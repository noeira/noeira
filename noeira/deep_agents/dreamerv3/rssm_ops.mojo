"""RSSM custom leaf ops — the only non-composable pieces of the RSSM `_core`.

SPIKE (design redesign): everything else in `_core` is existing nn modules
(`Sequential[Linear, RMSNorm, GELU]`, `BlockLinear`) wired by a
`ComputeGraph`. These three ops carry the algorithm-specific math (validated
in PR5b's `core_vjp`); the graph handles all grad routing/accumulation.

  * `ActionSquash[ACT]`         — a = action / sg(max(1, |action|))
  * `BlockGroupAssemble[D,H,G]` — per block g: [deter[g·dpb:+dpb], x0, x1, x2]
  * `GRUGate[D,G]`              — block-interleaved GRU gates + mix with deter
  * `StraightThroughSample[S,C]`— one-hot categorical sample, straight-through

Storage-surface port (off legacy `nn`): each follows the storage leaf pattern
(elementwise.mojo template): ARITY / IN_DIMS / OUT_DIM, `make[target,INIT]`,
`forward(inputs: TensorRefs, mut out: Tensor)`, `vjp(forward_input, grad_output,
grad_inputs)`. NO `TargetStorage`, NO cached-pointer fields — the storage `vjp`
receives `forward_input` (the SAME inputs the forward saw), so the backward
RECOMPUTES anything it needs (e.g. softmax) from `forward_input[i]`. No params →
inherits the no-op `for_each_param`/`zero_grad`.
"""

from std.math import tanh, exp
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile, tile


@always_inline
def _sig(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


# ──────────────────────────────────────────────────────────────────────
# GPU kernels for the custom RSSM ops. Top-level fns so `enqueue_function`
# can bind them. Each mirrors the op's CPU body. Math is identical to the
# CPU path → CPU↔GPU parity ≤1e-4 (per-op spikes). KEPT AS-IS from the
# legacy file — they take `LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]`
# (the GPU kernel ABI), now fed by `Tensor.lt["gpu", layout]()` at the
# launch sites.
# ──────────────────────────────────────────────────────────────────────


def _asq_fwd_kernel[N: Int](
    x: Tile[DT, N],
    o: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        var v = rebind[Scalar[DT]](x[i])
        var av = v if v >= Scalar[DT](0.0) else -v
        var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
        o[i] = v / denom


def _asq_bwd_kernel[N: Int](
    x: Tile[DT, N],
    go: Tile[DT, N],
    gi: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        var v = rebind[Scalar[DT]](x[i])
        var av = v if v >= Scalar[DT](0.0) else -v
        var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
        gi[i] = rebind[Scalar[DT]](go[i]) / denom


@always_inline
def _sigk(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


# ── GRUGate: N=B·D, D=DETER, DPB=DETER//BLOCKS. gru [B,3D], deter/nd [B,D];
#    block grp uses gru lanes [grp·3·DPB : +3·DPB]. ──────────────────────
def _gru_fwd_kernel[N: Int, D: Int, DPB: Int](
    gru: Tile[DT, N * 3],
    deter: Tile[DT, N],
    nd: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        var b = i // D
        var d = i % D
        var grp = d // DPB
        var k = d % DPB
        var gb = b * (3 * D) + grp * (3 * DPB) + k
        var reset = _sigk(rebind[Scalar[DT]](gru[gb]))
        var cand = tanh(reset * rebind[Scalar[DT]](gru[gb + DPB]))
        var upd = _sigk(rebind[Scalar[DT]](gru[gb + 2 * DPB]) - Scalar[DT](1.0))
        nd[i] = upd * cand + (Scalar[DT](1.0) - upd) * rebind[Scalar[DT]](deter[i])


def _gru_bwd_kernel[N: Int, D: Int, DPB: Int](
    gru: Tile[DT, N * 3],
    deter: Tile[DT, N],
    go: Tile[DT, N],
    g_gru: Tile[DT, N * 3],
    g_deter: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        var b = i // D
        var d = i % D
        var grp = d // DPB
        var k = d % DPB
        var gb = b * (3 * D) + grp * (3 * DPB) + k
        var c_pre = rebind[Scalar[DT]](gru[gb + DPB])
        var reset = _sigk(rebind[Scalar[DT]](gru[gb]))
        var cand = tanh(reset * c_pre)
        var upd = _sigk(rebind[Scalar[DT]](gru[gb + 2 * DPB]) - Scalar[DT](1.0))
        var d_prev = rebind[Scalar[DT]](deter[i])
        var gov = rebind[Scalar[DT]](go[i])
        var d_update = gov * (cand - d_prev)
        var d_cand = gov * upd
        g_deter[i] = gov * (Scalar[DT](1.0) - upd)
        var d_m = d_cand * (Scalar[DT](1.0) - cand * cand)
        g_gru[gb] = d_m * c_pre * reset * (Scalar[DT](1.0) - reset)
        g_gru[gb + DPB] = d_m * reset
        g_gru[gb + 2 * DPB] = d_update * upd * (Scalar[DT](1.0) - upd)


# ── BlockGroupAssemble: OUT=BLOCKS·PG, PG=DPB+3H. fwd one thread/out-elem;
#    bwd split — g_deter per (B·D), g_x* per (B·H) reducing over groups. ──
def _bga_fwd_kernel[
    NO: Int, OUT: Int, D: Int, H: Int, DPB: Int, PG: Int
](
    deter: Tile[DT, NO],
    x0: Tile[DT, NO],
    x1: Tile[DT, NO],
    x2: Tile[DT, NO],
    o: Tile[DT, NO],
):
    var i = Int(global_idx.x)
    if i < NO:
        var b = i // OUT
        var ob = i % OUT
        var grp = ob // PG
        var local = ob % PG
        if local < DPB:
            o[i] = rebind[Scalar[DT]](deter[b * D + grp * DPB + local])
        elif local < DPB + H:
            o[i] = rebind[Scalar[DT]](x0[b * H + (local - DPB)])
        elif local < DPB + 2 * H:
            o[i] = rebind[Scalar[DT]](x1[b * H + (local - DPB - H)])
        else:
            o[i] = rebind[Scalar[DT]](x2[b * H + (local - DPB - 2 * H)])


def _bga_bwd_deter_kernel[
    ND: Int, GON: Int, OUT: Int, D: Int, DPB: Int, PG: Int
](
    go: Tile[DT, GON],
    g_deter: Tile[DT, ND],
):
    var i = Int(global_idx.x)
    if i < ND:
        var b = i // D
        var d = i % D
        var grp = d // DPB
        var k = d % DPB
        g_deter[i] = rebind[Scalar[DT]](go[b * OUT + grp * PG + k])


def _bga_bwd_x_kernel[
    NH: Int, GON: Int, OUT: Int, H: Int, DPB: Int, PG: Int, BLOCKS: Int
](
    go: Tile[DT, GON],
    gx0: Tile[DT, NH],
    gx1: Tile[DT, NH],
    gx2: Tile[DT, NH],
):
    var i = Int(global_idx.x)
    if i < NH:
        var b = i // H
        var h = i % H
        var s0: Scalar[DT] = 0.0
        var s1: Scalar[DT] = 0.0
        var s2: Scalar[DT] = 0.0
        for grp in range(BLOCKS):
            var base = b * OUT + grp * PG + DPB
            s0 += rebind[Scalar[DT]](go[base + h])
            s1 += rebind[Scalar[DT]](go[base + H + h])
            s2 += rebind[Scalar[DT]](go[base + 2 * H + h])
        gx0[i] = s0
        gx1[i] = s1
        gx2[i] = s2


# ── StraightThroughSample: per (b,s) group of C lanes — softmax + unimix +
#    PHILOX categorical sample (fwd, inverse-CDF); grad_z = (1-u)·sm·(go − Σ go·sm)
#    (bwd). The bwd RECOMPUTES softmax(z) from `forward_input[0]` (z) — no cached
#    `sm` field, and is INDEPENDENT of which class was sampled (straight-through),
#    so fwd sampling needs no special bwd handling.
#
#    The sample is the DreamerV3 stochastic latent (paper's discrete repr) — NOT
#    the deterministic mode. Randomness is device-resident Philox keyed by
#    `seed + group` with a per-call `offset` (advanced by `_st_off_inc_k` after
#    every forward) so successive timesteps/steps draw disjoint streams. Philox is
#    pure uint32→float32 → BIT-IDENTICAL on CPU and GPU for the same (seed,offset),
#    which is what keeps the CPU↔GPU parity gates exact (sampling is discontinuous;
#    a 1-ULP uniform diff would flip the class). The CPU forward mirrors this with
#    a host `off_h` counter advanced in lockstep. Capture-safe: `offset_buf` is a
#    stable device buffer (created in `make`, never recreated) and is read/advanced
#    purely device-side — the proven imagination-noise pattern.
def _st_sample_kernel[NG: Int, C: Int](
    z: Tile[DT, NG * C],
    o: Tile[DT, NG * C],
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
    unimix: Scalar[DT],
):
    var gg = Int(global_idx.x)
    if gg < NG:
        var base = gg * C
        # numerically-stable softmax(z[base:base+C]).
        var zmax = rebind[Scalar[DT]](z[base])
        for c in range(1, C):
            var zc = rebind[Scalar[DT]](z[base + c])
            if zc > zmax:
                zmax = zc
        var ssum = Scalar[DT](0.0)
        for c in range(C):
            ssum += exp(rebind[Scalar[DT]](z[base + c]) - zmax)
        var inv = Scalar[DT](1.0) / ssum
        var one_m_u = Scalar[DT](1.0) - unimix
        var uc = unimix / Scalar[DT](C)
        # Philox uniform in [0,1): per-group stream (seed+gg), per-call offset.
        var off = rebind[UInt64](offset_buf[0])
        var philox = PhiloxRandom(seed=seed + UInt64(gg), offset=off)
        var u01 = Scalar[DT](philox.step_uniform()[0])
        # inverse-CDF over the unimix-mixed probs → sampled class.
        var acc = Scalar[DT](0.0)
        var ksel = C - 1
        for c in range(C):
            var sm_c = exp(rebind[Scalar[DT]](z[base + c]) - zmax) * inv
            var p_c = one_m_u * sm_c + uc
            acc += p_c
            if u01 < acc:
                ksel = c
                break
        for c in range(C):
            o[base + c] = Scalar[DT](1.0) if c == ksel else Scalar[DT](0.0)


def _st_off_inc_k[STRIDE: Int](
    offset_buf: Tile[DType.uint64, 1],
):
    """Advance the device stoch-sample Philox offset by STRIDE (one thread).
    Capture-safe: pure device read-modify-write, no host sync."""
    if Int(global_idx.x) == 0:
        offset_buf[0] = offset_buf[0] + UInt64(STRIDE)


def _st_bwd_kernel[NG: Int, C: Int](
    z: Tile[DT, NG * C],
    go: Tile[DT, NG * C],
    gz: Tile[DT, NG * C],
    one_m_u: Scalar[DT],
):
    var gg = Int(global_idx.x)
    if gg < NG:
        var base = gg * C
        # Recompute softmax(z) (numerically-stable max-subtract).
        var zmax = rebind[Scalar[DT]](z[base])
        for c in range(1, C):
            var zc = rebind[Scalar[DT]](z[base + c])
            if zc > zmax:
                zmax = zc
        var ssum: Scalar[DT] = 0.0
        for c in range(C):
            ssum += exp(rebind[Scalar[DT]](z[base + c]) - zmax)
        var inv = Scalar[DT](1.0) / ssum
        var dot: Scalar[DT] = 0.0
        for c in range(C):
            var smc = exp(rebind[Scalar[DT]](z[base + c]) - zmax) * inv
            dot += rebind[Scalar[DT]](go[base + c]) * smc
        for c in range(C):
            var smc = exp(rebind[Scalar[DT]](z[base + c]) - zmax) * inv
            gz[base + c] = one_m_u * smc * (rebind[Scalar[DT]](go[base + c]) - dot)


# ──────────────────────────────────────────────────────────────────────
# ActionSquash[ACT] — a = action / sg(max(1, |action|))   (arity 1)
# ──────────────────────────────────────────────────────────────────────


struct ActionSquash[ACT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.ACT)
    comptime OUT_DIM = Self.ACT

    @staticmethod
    def display_label() -> String:
        return String("ActionSquash")

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.ACT
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            for i in range(N):
                var v = in0.data[i]
                var av = v if v >= Scalar[DT](0.0) else -v
                var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
                out.data[i] = v / denom
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_asq_fwd_kernel[N]](
                in0.tt["gpu", N](),
                out.tt["gpu", N](),
                grid_dim=nb,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.ACT
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            for i in range(N):
                var v = fin.data[i]
                var av = v if v >= Scalar[DT](0.0) else -v
                var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
                gin.data[i] = grad_output.data[i] / denom
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_asq_bwd_kernel[N]](
                fin.tt["gpu", N](),
                grad_output.tt["gpu", N](),
                gin.tt["gpu", N](),
                grid_dim=nb,
                block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# BlockGroupAssemble[DETER, H, BLOCKS] — build the dynhid input.
#   per block g: [ deter[g·dpb : +dpb] , x0 , x1 , x2 ]   (x* shared/repeated)
# ──────────────────────────────────────────────────────────────────────


struct BlockGroupAssemble[DETER: Int, H: Int, BLOCKS: Int](Module):
    comptime DPB = Self.DETER // Self.BLOCKS
    comptime PER_GROUP = Self.DPB + 3 * Self.H
    comptime ARITY: Int = 4
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = Self.BLOCKS * Self.PER_GROUP   # = DETER + 3·H·BLOCKS

    @staticmethod
    def display_label() -> String:
        return String("BlockGroupAssemble")

    @staticmethod
    def _mk_in_dims() -> Array[Int, 4]:
        var d = Array[Int, 4](fill=Self.H)
        d[0] = Self.DETER
        return d^

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime D = Self.DETER
        comptime HH = Self.H
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime pg = Self.PER_GROUP
        comptime OUT = Self.OUT_DIM
        ref deter = inputs[0]
        ref x0 = inputs[1]
        ref x1 = inputs[2]
        ref x2 = inputs[3]
        comptime if target == "cpu":
            out.ensure(B * OUT)
            for b in range(B):
                for grp in range(g):
                    var off = b * OUT + grp * pg
                    for k in range(dpb):
                        out.data[off + k] = deter.data[b * D + grp * dpb + k]
                    for k in range(HH):
                        out.data[off + dpb + k] = x0.data[b * HH + k]
                    for k in range(HH):
                        out.data[off + dpb + HH + k] = x1.data[b * HH + k]
                    for k in range(HH):
                        out.data[off + dpb + 2 * HH + k] = x2.data[b * HH + k]
        else:
            var c = ctx.value()
            comptime NO = B * OUT
            out.ensure_gpu(c, NO)
            comptime nb = (NO + TPB - 1) // TPB
            c.enqueue_function[_bga_fwd_kernel[NO, OUT, D, HH, dpb, pg]](
                deter.tt["gpu", NO](),
                x0.tt["gpu", NO](),
                x1.tt["gpu", NO](),
                x2.tt["gpu", NO](),
                out.tt["gpu", NO](),
                grid_dim=nb,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime D = Self.DETER
        comptime HH = Self.H
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime pg = Self.PER_GROUP
        comptime OUT = Self.OUT_DIM
        ref g_deter = grad_inputs[0]
        ref g_x0 = grad_inputs[1]
        ref g_x1 = grad_inputs[2]
        ref g_x2 = grad_inputs[3]
        comptime if target == "cpu":
            g_deter.ensure(B * D)
            g_x0.ensure(B * HH)
            g_x1.ensure(B * HH)
            g_x2.ensure(B * HH)
            for i in range(B * HH):
                g_x0.data[i] = 0.0
                g_x1.data[i] = 0.0
                g_x2.data[i] = 0.0
            for b in range(B):
                for grp in range(g):
                    var off = b * OUT + grp * pg
                    for k in range(dpb):
                        g_deter.data[b * D + grp * dpb + k] = grad_output.data[
                            off + k
                        ]
                    for k in range(HH):
                        g_x0.data[b * HH + k] += grad_output.data[off + dpb + k]
                        g_x1.data[b * HH + k] += grad_output.data[
                            off + dpb + HH + k
                        ]
                        g_x2.data[b * HH + k] += grad_output.data[
                            off + dpb + 2 * HH + k
                        ]
        else:
            var c = ctx.value()
            comptime GON = B * OUT
            comptime ND = B * D
            comptime NH = B * HH
            g_deter.ensure_gpu(c, ND)
            g_x0.ensure_gpu(c, NH)
            g_x1.ensure_gpu(c, NH)
            g_x2.ensure_gpu(c, NH)
            comptime nbd = (ND + TPB - 1) // TPB
            c.enqueue_function[_bga_bwd_deter_kernel[ND, GON, OUT, D, dpb, pg]](
                grad_output.tt["gpu", GON](),
                g_deter.tt["gpu", ND](),
                grid_dim=nbd,
                block_dim=TPB,
            )
            comptime nbh = (NH + TPB - 1) // TPB
            c.enqueue_function[
                _bga_bwd_x_kernel[NH, GON, OUT, HH, dpb, pg, g]
            ](
                grad_output.tt["gpu", GON](),
                g_x0.tt["gpu", NH](),
                g_x1.tt["gpu", NH](),
                g_x2.tt["gpu", NH](),
                grid_dim=nbh,
                block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# GRUGate[DETER, BLOCKS] — block-interleaved GRU gates + mix with deter.
#   in0 = gru [B, 3·DETER]   in1 = deter [B, DETER]   → new_deter [B, DETER]
# ──────────────────────────────────────────────────────────────────────


struct GRUGate[DETER: Int, BLOCKS: Int](Module):
    comptime DPB = Self.DETER // Self.BLOCKS
    comptime OPB = 3 * Self.DPB
    comptime GRU_DIM = 3 * Self.DETER
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = Self.DETER

    @staticmethod
    def display_label() -> String:
        return String("GRUGate")

    @staticmethod
    def _mk_in_dims() -> Array[Int, 2]:
        var d = Array[Int, 2](fill=Self.DETER)
        d[0] = Self.GRU_DIM
        return d^

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime D = Self.DETER
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime opb = Self.OPB
        comptime GD = Self.GRU_DIM
        ref gru = inputs[0]
        ref deter = inputs[1]
        comptime if target == "cpu":
            out.ensure(B * D)
            for b in range(B):
                for grp in range(g):
                    var gb = b * GD + grp * opb
                    var dbase = b * D + grp * dpb
                    for k in range(dpb):
                        var reset = _sig(gru.data[gb + k])
                        var cand = tanh(reset * gru.data[gb + dpb + k])
                        var upd = _sig(gru.data[gb + 2 * dpb + k] - Scalar[DT](1.0))
                        out.data[dbase + k] = upd * cand + (
                            Scalar[DT](1.0) - upd
                        ) * deter.data[dbase + k]
        else:
            var c = ctx.value()
            comptime N = B * D
            out.ensure_gpu(c, N)
            comptime nb = (N + TPB - 1) // TPB
            # gru is [B, 3D] = N*3 elements; deter/out are [B, D] = N.
            c.enqueue_function[_gru_fwd_kernel[N, D, dpb]](
                gru.tt["gpu", N * 3](),
                deter.tt["gpu", N](),
                out.tt["gpu", N](),
                grid_dim=nb,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime D = Self.DETER
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime opb = Self.OPB
        comptime GD = Self.GRU_DIM
        ref gru = forward_input[0]
        ref deter = forward_input[1]
        ref g_gru = grad_inputs[0]
        ref g_deter = grad_inputs[1]
        comptime if target == "cpu":
            g_gru.ensure(B * GD)
            g_deter.ensure(B * D)
            for i in range(B * GD):
                g_gru.data[i] = 0.0
            for b in range(B):
                for grp in range(g):
                    var gb = b * GD + grp * opb
                    var dbase = b * D + grp * dpb
                    for k in range(dpb):
                        var r_pre = gru.data[gb + k]
                        var c_pre = gru.data[gb + dpb + k]
                        var u_pre = gru.data[gb + 2 * dpb + k] - Scalar[DT](1.0)
                        var reset = _sig(r_pre)
                        var cand = tanh(reset * c_pre)
                        var upd = _sig(u_pre)
                        var d_prev = deter.data[dbase + k]
                        var gov = grad_output.data[dbase + k]
                        var d_update = gov * (cand - d_prev)
                        var d_cand = gov * upd
                        g_deter.data[dbase + k] = gov * (Scalar[DT](1.0) - upd)
                        var d_m = d_cand * (Scalar[DT](1.0) - cand * cand)
                        g_gru.data[gb + k] = d_m * c_pre * reset * (
                            Scalar[DT](1.0) - reset
                        )
                        g_gru.data[gb + dpb + k] = d_m * reset
                        g_gru.data[gb + 2 * dpb + k] = d_update * upd * (
                            Scalar[DT](1.0) - upd
                        )
        else:
            var c = ctx.value()
            comptime N = B * D
            g_gru.ensure_gpu(c, N * 3)
            g_deter.ensure_gpu(c, N)
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_gru_bwd_kernel[N, D, dpb]](
                gru.tt["gpu", N * 3](),
                deter.tt["gpu", N](),
                grad_output.tt["gpu", N](),
                g_gru.tt["gpu", N * 3](),
                g_deter.tt["gpu", N](),
                grid_dim=nb,
                block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# StraightThroughSample[STOCH, CLASSES] — one-hot categorical sample with
# straight-through gradient.  logits[B, S·C] → onehot[B, S·C].
#   forward : value = onehot(sample)            (per (b,s) over CLASSES)
#   backward: grad_z = (1-u)·softmax_vjp(z)      (INDEPENDENT of the sample —
#             the index is stop-grad'd; grad flows through the unimix probs)
#
# Forward currently takes the argmax (deterministic placeholder); the trainer
# wires a PhiloxRandom categorical sample. The backward — the trainable path —
# is exact and validated (st_fixture). The storage vjp RECOMPUTES softmax(z)
# from `forward_input[0]` (no cached `sm`), bit-identical to the cached path.
# ──────────────────────────────────────────────────────────────────────


struct StraightThroughSample[STOCH: Int, CLASSES: Int](Module):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.SC)
    comptime OUT_DIM = Self.SC

    @staticmethod
    def display_label() -> String:
        return String("STSample")

    var unimix: Scalar[DT]   # real hyperparam (unimix prob), NOT cache
    # Philox stochastic-sample state (fixed base seed; per-call offset).
    var seed: UInt64
    var off_h: UInt64                                   # host offset (CPU path)
    var off_d: Optional[DeviceBuffer[DType.uint64]]     # device offset (GPU path)

    def __init__(out self):
        self.unimix = Scalar[DT](0.01)
        self.seed = UInt64(0x57BE_AC10_5A33_71E5)
        self.off_h = UInt64(0)
        self.off_d = None

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        # Create the device offset buffer at graph-build time (capture-safe: a
        # stable buffer never recreated, advanced device-side per forward).
        comptime if target == "gpu":
            var b = ctx.value().enqueue_create_buffer[DType.uint64](1)
            b.enqueue_fill(UInt64(0))
            s.off_d = b^
        return s^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime C = Self.CLASSES
        comptime NG = B * Self.STOCH       # group count = offset STRIDE
        ref z = inputs[0]
        var one_m_u = Scalar[DT](1.0) - self.unimix
        var uc = self.unimix / Scalar[DT](C)
        comptime if target == "cpu":
            out.ensure(B * Self.SC)
            for b in range(B):
                for s in range(Self.STOCH):
                    var gg = b * Self.STOCH + s
                    var base = gg * C
                    # numerically-stable softmax(z[base:base+C]).
                    var zmax = z.data[base]
                    for c in range(1, C):
                        if z.data[base + c] > zmax:
                            zmax = z.data[base + c]
                    var ssum = Scalar[DT](0.0)
                    for c in range(C):
                        ssum += exp(z.data[base + c] - zmax)
                    var inv = Scalar[DT](1.0) / ssum
                    # Philox uniform: SAME (seed+gg, offset) as the GPU kernel →
                    # bit-identical sample (CPU↔GPU parity).
                    var philox = PhiloxRandom(
                        seed=self.seed + UInt64(gg), offset=self.off_h
                    )
                    var u01 = Scalar[DT](philox.step_uniform()[0])
                    var acc = Scalar[DT](0.0)
                    var ksel = C - 1
                    for c in range(C):
                        var sm_c = exp(z.data[base + c] - zmax) * inv
                        var p_c = one_m_u * sm_c + uc
                        acc += p_c
                        if u01 < acc:
                            ksel = c
                            break
                    for c in range(C):
                        out.data[base + c] = (
                            Scalar[DT](1.0) if c == ksel else Scalar[DT](0.0)
                        )
            self.off_h = self.off_h + UInt64(NG)
        else:
            var c = ctx.value()
            comptime NN = B * Self.SC
            out.ensure_gpu(c, NN)
            comptime nb = (NG + TPB - 1) // TPB
            # Lazy-create the offset buffer for direct-constructed instances
            # (unit tests). Production builds create it in `make` BEFORE any
            # capture, so this branch never runs inside a captured region.
            if not self.off_d:
                var ob = c.enqueue_create_buffer[DType.uint64](1)
                ob.enqueue_fill(UInt64(0))
                self.off_d = ob^
            var off_lt = tile[1](self.off_d.value())
            c.enqueue_function[_st_sample_kernel[NG, C]](
                z.tt["gpu", NN](),
                out.tt["gpu", NN](),
                self.seed, off_lt, self.unimix,
                grid_dim=nb,
                block_dim=TPB,
            )
            c.enqueue_function[_st_off_inc_k[NG]](
                off_lt, grid_dim=1, block_dim=1,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime C = Self.CLASSES
        ref z = forward_input[0]
        ref gz = grad_inputs[0]
        var one_m_u = Scalar[DT](1.0) - self.unimix
        comptime if target == "cpu":
            gz.ensure(B * Self.SC)
            for b in range(B):
                for s in range(Self.STOCH):
                    var base = (b * Self.STOCH + s) * C
                    # Recompute softmax(z) (numerically-stable max-subtract).
                    var zmax = z.data[base]
                    for c in range(1, C):
                        if z.data[base + c] > zmax:
                            zmax = z.data[base + c]
                    var ssum: Scalar[DT] = 0.0
                    for c in range(C):
                        ssum += exp(z.data[base + c] - zmax)
                    var inv = Scalar[DT](1.0) / ssum
                    var dot: Scalar[DT] = 0.0
                    for c in range(C):
                        var smc = exp(z.data[base + c] - zmax) * inv
                        dot += grad_output.data[base + c] * smc
                    for c in range(C):
                        var smc = exp(z.data[base + c] - zmax) * inv
                        gz.data[base + c] = one_m_u * smc * (
                            grad_output.data[base + c] - dot
                        )
        else:
            var c = ctx.value()
            comptime NN = B * Self.SC
            comptime NG = B * Self.STOCH
            gz.ensure_gpu(c, NN)
            comptime nb = (NG + TPB - 1) // TPB
            c.enqueue_function[_st_bwd_kernel[NG, C]](
                z.tt["gpu", NN](),
                grad_output.tt["gpu", NN](),
                gz.tt["gpu", NN](),
                one_m_u,
                grid_dim=nb,
                block_dim=TPB,
            )
