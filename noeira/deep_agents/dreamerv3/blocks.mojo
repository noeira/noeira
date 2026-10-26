"""DreamerV3 trainer blocks — SAC-style composable make/step units (storage).

Mirrors `deep_agents/tdmpc2/wm_step.mojo`: each block is a concrete struct with
`make[target](ctx)` + `step[target](mut state, mut <modules/opts>)`.
`target: StaticString` is comptime ("cpu"/"gpu"); `ctx: Optional[DeviceContext]`
is threaded at runtime via `DreamerState`. The CPU/GPU split inside each step is
a `comptime if target == "cpu": ... else: ...`.

Storage migration (Stage: DreamerV3 BPTT/imagination driver): inputs/scratch are
storage `Tensor`s (`.data` host List / `.dev` Optional[DeviceBuffer]). Persistent
scratch is allocated ONCE in `make[target]` via `_mk[target](n, ctx)` and
reused every step. Module nets (enc / value / policy heads) drive through the
storage `Module` surface (`forward[target,B](TensorRefs[1](in), out, ctx)` /
`vjp[target,B](TensorRefs[1](fin), go, TensorRefs[1](gi), ctx)`); the loss graphs
(WMCore/WMImagine/Dec/Rew/Con) own their params and drive through the storage
`ComputeGraph` surface (`set_input` / `forward[B,target]` / `vjp[B,target]` /
`node_output` / `grad_input`); DreamerOpt drives Modules via `step[target,M]` and
graphs via `begin_step(); walk_params[target](graph, opt, ctx)`.

The Phase-1 host loss helpers (`imag_loss_*` / `repl_loss_backward` /
`twohot_pred` / `bounded_std` / `cat_sample`) are unchanged free functions taking
raw `Pointer[Scalar[DT], MutAnyOrigin]`; CPU call sites pass
`rebind[...](tensor.data.unsafe_ptr())` (sanctioned host-pointer use).

GPU paths: the CPU branch is the convergence-gated reference. `_wm_gpu` runs the
WM-BPTT scan on `.dev` but marshals the per-step reset masks / carries / head
losses through host `.data`. The GPU AC has two flavours, both device-resident:
CONTINUOUS (`_ac_gpu_cont`) and DISCRETE (`_ac_gpu_disc`) — the imagination rollout,
λ-return, imag/repl loss and gradients all run through the device kernels in this
file ([NS,TI,W] histories), with the only host round-trips being a one-time
noise/bins upload, ONE `ret` download for the percentile retnorm, and the
`want_diag`-gated diagnostic readout. All paths mirror `_ac_cpu`/`_wm_cpu` so the
CPU↔GPU parity gates hold.
"""

from noeira.nn.core.param import walk_params
from std.memory import alloc
from std.math import tanh, exp, sqrt, log, cos
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from layout import TileTensor, row_major
from noeira.core.tiles import Tile, tile
from max.gpu import global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from noeira.nn.primitives.ops.swish_op import SwishOp
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs, child_refs
from noeira.deep_agents.dreamerv3.twohot import twohot_pred
from noeira.deep_agents.dreamerv3.dists import bounded_std
from noeira.deep_agents.dreamerv3.dists_discrete import cat_sample, UNIMIX
from noeira.deep_agents.dreamerv3.normalize import PercentileNormalize
from noeira.deep_agents.dreamerv3.repl_loss import repl_loss_backward
from noeira.deep_agents.dreamerv3.imag_loss import (
    imag_loss_cpu, imag_loss_backward,
)
from noeira.deep_agents.dreamerv3.param_sync import (
    collect_graph_params, apply_graph_params,
    collect_graph_params_device, apply_graph_params_device,
)
from noeira.deep_agents.dreamerv3.polyak import polyak_module
from noeira.deep_agents.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from noeira.deep_agents.dreamerv3.nets import (
    DreamerEncoder, DreamerDecoder, DreamerValue, DreamerPolicyHead,
)
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer, Zero
from noeira.nn.optimizer.dreamer_opt import DreamerOpt


@always_inline
def _hp(mut t: Tensor) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Sanctioned host-pointer view of a storage Tensor's CPU `data` — for the
    raw-pointer Phase-1 loss helpers (CPU only)."""
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](t.data.unsafe_ptr())


@always_inline
def _mk[target: StaticString](
    n: Int, ctx: Optional[DeviceContext]
) raises -> Tensor:
    """Scratch allocator for the DreamerV3 blocks. On CPU == `Tensor.make`. On
    GPU it allocates the device buffer AND sizes the host `.data` List, because
    the GPU WM/AC paths marshal the per-step reset masks / carries / loss reads
    through host `.data` (upload/download of small windows) — so the host side
    must be pre-sized to be indexed before the first download."""
    var t = Tensor.make[target](n, ctx)
    comptime if target == "gpu":
        t.ensure(n)  # size the host `.data` List (device buffer kept)
    return t^


# ── GPU marshalling kernels (kept for the GPU port; `def` not `fn`) ──────


def _bcopy[N: Int](
    src: Tile[DT, N],
    dst: Tile[DT, N],
):
    """Contiguous device→device copy (carry / grad threading)."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


def _seed_asm_k[B_: Int, CARRY_: Int, D_: Int, SC_: Int](
    seed: Tile[DT, B_ * CARRY_],
    gcd: Tile[DT, B_ * D_],
    gcs: Tile[DT, B_ * SC_],
    dnd: Tile[DT, B_ * D_],
    rnd: Tile[DT, B_ * D_],
    cnd: Tile[DT, B_ * D_],
    dsn: Tile[DT, B_ * SC_],
    rsn: Tile[DT, B_ * SC_],
    csn: Tile[DT, B_ * SC_],
    dyn: Scalar[DT],
    rep: Scalar[DT],
):
    """Assemble the BPTT carry seed for one batch row. One thread per element
    over B*max(D,SC) (was B-parallel); the 2 (dyn,rep) scalars on k==0."""
    comptime MW = D_ if D_ > SC_ else SC_
    var i = Int(global_idx.x)
    if i < B_ * MW:
        var b = i // MW
        var k = i % MW
        if k == 0:
            seed[b * CARRY_] = dyn
            seed[b * CARRY_ + 1] = rep
        if k < D_:
            seed[b * CARRY_ + 2 + k] = (
                rebind[Scalar[DT]](gcd[b * D_ + k])
                + rebind[Scalar[DT]](dnd[b * D_ + k])
                + rebind[Scalar[DT]](rnd[b * D_ + k])
                + rebind[Scalar[DT]](cnd[b * D_ + k])
            )
        if k < SC_:
            seed[b * CARRY_ + 2 + D_ + k] = (
                rebind[Scalar[DT]](gcs[b * SC_ + k])
                + rebind[Scalar[DT]](dsn[b * SC_ + k])
                + rebind[Scalar[DT]](rsn[b * SC_ + k])
                + rebind[Scalar[DT]](csn[b * SC_ + k])
            )


def _feat_concat_k[B_: Int, D_: Int, SC_: Int](
    deter: Tile[DT, B_ * D_],
    stoch: Tile[DT, B_ * SC_],
    feat: Tile[DT, B_ * (D_ + SC_)],
):
    """feat[b] = concat(deter[b], stoch[b])  (FEAT = D + SC). One thread per
    output element over B*(D+SC) (was B-parallel with a serial FEAT loop → a
    single block at NS=B=256)."""
    var F = D_ + SC_
    var i = Int(global_idx.x)
    if i < B_ * F:
        var b = i // F
        var k = i % F
        if k < D_:
            feat[i] = rebind[Scalar[DT]](deter[b * D_ + k])
        else:
            feat[i] = rebind[Scalar[DT]](stoch[b * SC_ + (k - D_)])


def _rowscale_k[B_: Int, W_: Int](
    src: Tile[DT, B_ * W_],
    dst: Tile[DT, B_ * W_],
    m: Tile[DT, B_],
):
    var i = Int(global_idx.x)
    if i < B_ * W_:
        dst[i] = rebind[Scalar[DT]](m[i // W_]) * rebind[Scalar[DT]](src[i])


def _rowscale_inplace_k[B_: Int, W_: Int](
    buf: Tile[DT, B_ * W_],
    m: Tile[DT, B_],
):
    var i = Int(global_idx.x)
    if i < B_ * W_:
        buf[i] = rebind[Scalar[DT]](m[i // W_]) * rebind[Scalar[DT]](buf[i])


# ──────────────────────────────────────────────────────────────────────
# Device-resident discrete imagination-AC kernels (Stage 1: move the
# λ-return / imag-loss / sampling off host). Histories are [NS, TI, W]
# (per-start contiguous over the rollout step `t`). Every kernel writes a
# DISJOINT slice per thread (one start `b`, or one (start,t) cell) — no
# cross-thread read-modify-write, so the NVIDIA reduction store-drop
# footgun does not apply. Mirrors the `_ac_cpu` math exactly so the
# CPU↔GPU parity gate holds. DISCRETE (unimix categorical) only.
# ──────────────────────────────────────────────────────────────────────


@always_inline
def _dev_twp[BINS_: Int, NL: Int, NB: Int](
    logits: Tile[DT, NL],
    base: Int,
    bins: Tile[DT, NB],
) -> Scalar[DT]:
    """Σ_c softmax(logits[base:base+BINS])_c · bins_c (device twohot_pred)."""
    var zmax = rebind[Scalar[DT]](logits[base])
    for c in range(1, BINS_):
        var v = rebind[Scalar[DT]](logits[base + c])
        if v > zmax:
            zmax = v
    var ssum = Scalar[DT](0.0)
    for c in range(BINS_):
        ssum += exp(rebind[Scalar[DT]](logits[base + c]) - zmax)
    var inv = Scalar[DT](1.0) / ssum
    var acc = Scalar[DT](0.0)
    for c in range(BINS_):
        acc += (
            exp(rebind[Scalar[DT]](logits[base + c]) - zmax)
            * inv
            * rebind[Scalar[DT]](bins[c])
        )
    return acc


@always_inline
def _dev_twohot_ce[BINS_: Int, NL: Int, NB: Int](
    logits: Tile[DT, NL],
    base: Int,
    bins: Tile[DT, NB],
    target: Scalar[DT],
) -> Scalar[DT]:
    """Twohot cross-entropy of `target` vs logits[base:] (device twohot_loss)."""
    var n_le = 0
    for c in range(BINS_):
        if rebind[Scalar[DT]](bins[c]) <= target:
            n_le += 1
    var below = n_le - 1
    var above = n_le
    if below < 0:
        below = 0
    if below > BINS_ - 1:
        below = BINS_ - 1
    if above < 0:
        above = 0
    if above > BINS_ - 1:
        above = BINS_ - 1
    var w_below: Scalar[DT]
    var w_above: Scalar[DT]
    if below == above:
        w_below = Scalar[DT](0.5)
        w_above = Scalar[DT](0.5)
    else:
        var db = rebind[Scalar[DT]](bins[below]) - target
        var da = rebind[Scalar[DT]](bins[above]) - target
        db = db if db >= Scalar[DT](0.0) else -db
        da = da if da >= Scalar[DT](0.0) else -da
        var tot = db + da
        w_below = da / tot
        w_above = db / tot
    var zmax = rebind[Scalar[DT]](logits[base])
    for c in range(1, BINS_):
        var v = rebind[Scalar[DT]](logits[base + c])
        if v > zmax:
            zmax = v
    var ssum = Scalar[DT](0.0)
    for c in range(BINS_):
        ssum += exp(rebind[Scalar[DT]](logits[base + c]) - zmax)
    var lse = zmax + log(ssum)
    var lp_below = rebind[Scalar[DT]](logits[base + below]) - lse
    var lp_above = rebind[Scalar[DT]](logits[base + above]) - lse
    return -(w_below * lp_below + w_above * lp_above)


@always_inline
def _dev_twohot_ce_bwd[BINS_: Int, NL: Int, NB: Int, NG: Int](
    logits: Tile[DT, NL],
    base: Int,
    bins: Tile[DT, NB],
    target: Scalar[DT],
    upstream: Scalar[DT],
    grad: Tile[DT, NG],
):
    """Accumulate upstream·(softmax − twohot(target)) into grad[base:base+BINS]
    (device twohot_loss_backward; grad slice must be pre-zeroed)."""
    var n_le = 0
    for c in range(BINS_):
        if rebind[Scalar[DT]](bins[c]) <= target:
            n_le += 1
    var below = n_le - 1
    var above = n_le
    if below < 0:
        below = 0
    if below > BINS_ - 1:
        below = BINS_ - 1
    if above < 0:
        above = 0
    if above > BINS_ - 1:
        above = BINS_ - 1
    var w_below: Scalar[DT]
    var w_above: Scalar[DT]
    if below == above:
        w_below = Scalar[DT](0.5)
        w_above = Scalar[DT](0.5)
    else:
        var db = rebind[Scalar[DT]](bins[below]) - target
        var da = rebind[Scalar[DT]](bins[above]) - target
        db = db if db >= Scalar[DT](0.0) else -db
        da = da if da >= Scalar[DT](0.0) else -da
        var tot = db + da
        w_below = da / tot
        w_above = db / tot
    var zmax = rebind[Scalar[DT]](logits[base])
    for c in range(1, BINS_):
        var v = rebind[Scalar[DT]](logits[base + c])
        if v > zmax:
            zmax = v
    var ssum = Scalar[DT](0.0)
    for c in range(BINS_):
        ssum += exp(rebind[Scalar[DT]](logits[base + c]) - zmax)
    var inv = Scalar[DT](1.0) / ssum
    for c in range(BINS_):
        grad[base + c] = rebind[Scalar[DT]](grad[base + c]) + upstream * (
            exp(rebind[Scalar[DT]](logits[base + c]) - zmax) * inv
        )
    grad[base + below] = rebind[Scalar[DT]](grad[base + below]) - upstream * w_below
    grad[base + above] = rebind[Scalar[DT]](grad[base + above]) - upstream * w_above


def _hist_store_k[W_: Int, TI_: Int, NS_: Int](
    src: Tile[DT, NS_ * W_],
    hist: Tile[DT, NS_ * TI_ * W_],
    t_arg: Int64,
):
    """Scatter the per-step output src[NS,W] into hist[NS,TI,W] at step t. One
    thread per element over NS*W (was NS-parallel with a serial W loop)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var i = Int(global_idx.x)
    if i < NS_ * W_:
        var b = i // W_
        var k = i % W_
        hist[(b * TI_ + t) * W_ + k] = rebind[Scalar[DT]](src[i])


def _hist_load_k[W_: Int, TI_: Int, NS_: Int](
    hist: Tile[DT, NS_ * TI_ * W_],
    dst: Tile[DT, NS_ * W_],
    t_arg: Int64,
):
    """Gather hist[NS,TI,W] step t into the contiguous dst[NS,W]. One thread per
    element over NS*W (was NS-parallel with a serial W loop)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var i = Int(global_idx.x)
    if i < NS_ * W_:
        var b = i // W_
        var k = i % W_
        dst[i] = rebind[Scalar[DT]](hist[(b * TI_ + t) * W_ + k])


def _hist_load_pol2_k[ACT_: Int, TI_: Int, NS_: Int](
    gpm: Tile[DT, NS_ * TI_ * ACT_],
    gps: Tile[DT, NS_ * TI_ * ACT_],
    dst: Tile[DT, NS_ * 2 * ACT_],
    t_arg: Int64,
):
    """Continuous policy cotangent: gather step-t gpmean → dst[:, 0:ACT] and
    gpstd → dst[:, ACT:2ACT] (the [mean_raw | std_raw] layout the 2·ACT policy
    head emits) for the per-step policy.vjp. One thread per element over
    NS*2*ACT (was NS-parallel with a serial ACT loop)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var i = Int(global_idx.x)
    if i < NS_ * 2 * ACT_:
        var b = i // (2 * ACT_)
        var j = i % (2 * ACT_)
        if j < ACT_:
            dst[i] = rebind[Scalar[DT]](gpm[(b * TI_ + t) * ACT_ + j])
        else:
            dst[i] = rebind[Scalar[DT]](gps[(b * TI_ + t) * ACT_ + (j - ACT_)])


def _noise_philox_k[N_: Int](
    dst: Tile[DT, N_],
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    """Device imagination noise: element i ← uniform[-1,1] via Philox(seed+i,
    offset). Replaces the host `random_float64` gen + H2D upload (Stage 3 P4
    follow-up) — pure device work, runs in the eager prologue each step. The
    offset lives in `offset_buf` (advanced by `_noise_off_inc_k`) so successive
    steps draw disjoint streams. Cast `step_uniform()[0]` straight to `Scalar[DT]`
    (the proven dropout/replay GPU idiom — an intermediate `Float32(...)` over the
    fp64 draw drags an fp64 op into the kernel, unsupported on Metal)."""
    var i = Int(global_idx.x)
    if i >= N_:
        return
    var off = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=off)
    var u = Scalar[DT](philox.step_uniform()[0])
    dst[i] = u * Scalar[DT](2.0) - Scalar[DT](1.0)


def _noise_off_inc_k[STRIDE: Int](
    offset_buf: Tile[DType.uint64, 1],
):
    """Advance the device noise offset by STRIDE (one thread)."""
    if Int(global_idx.x) == 0:
        offset_buf[0] = offset_buf[0] + UInt64(STRIDE)


def _cat_sample_hist_k[C_: Int, TI_: Int, NS_: Int](
    pb: Tile[DT, NS_ * C_],
    noise: Tile[DT, TI_ * NS_ * C_],
    at: Tile[DT, NS_ * C_],
    pmean_h: Tile[DT, NS_ * TI_ * C_],
    acts_h: Tile[DT, NS_ * TI_ * C_],
    u: Scalar[DT],
    t_arg: Int64,
):
    """Unimix-categorical sample (inverse-CDF) from pb[NS,C] using noise; write
    logits→pmean_h, one-hot→acts_h (both [NS,TI,C]) and the action fed to
    imagine→at. Mirrors `_ac_cpu` discrete branch (k = cat_sample)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var b = Int(global_idx.x)
    if b < NS_:
        var mx = rebind[Scalar[DT]](pb[b * C_])
        for c in range(1, C_):
            var v = rebind[Scalar[DT]](pb[b * C_ + c])
            if v > mx:
                mx = v
        var s = Scalar[DT](0.0)
        for c in range(C_):
            s += exp(rebind[Scalar[DT]](pb[b * C_ + c]) - mx)
        var inv = Scalar[DT](1.0) / s
        var one_m_u = Scalar[DT](1.0) - u
        var uc = u / Scalar[DT](C_)
        var z0 = rebind[Scalar[DT]](noise[(t * NS_ + b) * C_ + 0])
        var u01 = (z0 + Scalar[DT](1.0)) * Scalar[DT](0.5)
        var acc = Scalar[DT](0.0)
        var ksel = C_ - 1
        for c in range(C_):
            var sm_c = exp(rebind[Scalar[DT]](pb[b * C_ + c]) - mx) * inv
            var p_c = one_m_u * sm_c + uc
            acc += p_c
            if u01 < acc:
                ksel = c
                break
        for a in range(C_):
            var lg = rebind[Scalar[DT]](pb[b * C_ + a])
            pmean_h[(b * TI_ + t) * C_ + a] = lg
            var oh = Scalar[DT](1.0) if a == ksel else Scalar[DT](0.0)
            acts_h[(b * TI_ + t) * C_ + a] = oh
            at[b * C_ + a] = oh


def _gaussian_sample_hist_k[ACT_: Int, TI_: Int, NS_: Int](
    pb: Tile[DT, NS_ * 2 * ACT_],
    noise: Tile[DT, TI_ * NS_ * ACT_],
    at: Tile[DT, NS_ * ACT_],
    pmean_h: Tile[DT, NS_ * TI_ * ACT_],
    pstd_h: Tile[DT, NS_ * TI_ * ACT_],
    acts_h: Tile[DT, NS_ * TI_ * ACT_],
    minstd: Scalar[DT],
    maxstd: Scalar[DT],
    t_arg: Int64,
):
    """Bounded-normal reparameterized sample (device) for the CONTINUOUS AC
    imagination rollout — the Gaussian counterpart of `_cat_sample_hist_k`.
    Per start b, per action dim a: mean_raw = pb[b, a], std_raw = pb[b, ACT+a];
        std = (maxstd-minstd)·sigmoid(std_raw+2) + minstd
        act = tanh(mean_raw) + std · noise[t, b, a]
    writes mean_raw→pmean_h, std_raw→pstd_h, act→acts_h (all [NS,TI,ACT]) and the
    action fed to imagine→at. Mirrors the `_ac_gpu`/`_ac_cpu` host sampling
    bit-for-bit (bounded_std inlined; all Scalar[DT] so Metal-safe — no fp64)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var b = Int(global_idx.x)
    if b < NS_:
        for a in range(ACT_):
            var mr = rebind[Scalar[DT]](pb[b * 2 * ACT_ + a])
            var sr = rebind[Scalar[DT]](pb[b * 2 * ACT_ + ACT_ + a])
            pmean_h[(b * TI_ + t) * ACT_ + a] = mr
            pstd_h[(b * TI_ + t) * ACT_ + a] = sr
            var z = rebind[Scalar[DT]](noise[(t * NS_ + b) * ACT_ + a])
            var sig = Scalar[DT](1.0) / (
                Scalar[DT](1.0) + exp(-(sr + Scalar[DT](2.0)))
            )
            var std = (maxstd - minstd) * sig + minstd
            var smp = tanh(mr) + std * z
            acts_h[(b * TI_ + t) * ACT_ + a] = smp
            at[b * ACT_ + a] = smp


def _rewconv_hist_k[BINS_: Int, TI_: Int, NS_: Int](
    rew_logits: Tile[DT, NS_ * BINS_],
    con_logit: Tile[DT, NS_],
    bins: Tile[DT, BINS_],
    rewv_h: Tile[DT, NS_ * TI_],
    conv_h: Tile[DT, NS_ * TI_],
    t_arg: Int64,
):
    """rewv = twohot_pred(rew logits); conv = sigmoid(con logit) → histories."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var b = Int(global_idx.x)
    if b < NS_:
        rewv_h[b * TI_ + t] = _dev_twp[BINS_](rew_logits, b * BINS_, bins)
        var cl = rebind[Scalar[DT]](con_logit[b])
        conv_h[b * TI_ + t] = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-cl))


def _imag_ret_k[NS_: Int, TI_: Int, BINS_: Int](
    vlog: Tile[DT, NS_ * TI_ * BINS_],
    svlog: Tile[DT, NS_ * TI_ * BINS_],
    rewv: Tile[DT, NS_ * TI_],
    conv: Tile[DT, NS_ * TI_],
    bins: Tile[DT, BINS_],
    ret: Tile[DT, NS_ * (TI_ - 1)],
    lam: Scalar[DT],
    # ⚠ Int32, NOT Int — `Int`/`UInt` are not `DevicePassable`. A bare
    # `Int` still compiles in `pixi run build` and fails only where the
    # kernel is LAUNCHED; keep the `Int32(...)` casts at the call sites.
    slowtar: Int32,  # 1 → bootstrap from svlog (EMA slowvalue); 0 → vlog
):
    """λ-return over the imagined rollout (disc=1) → ret[NS,TM1]. `slowtar`
    bootstraps from the EMA slowvalue (svlog) instead of the online value (vlog)
    — matches `imag_loss_cpu`'s slowtar branch. Per-start downward scan.
    (`slowtar` is a flag, not a Bool: neither `Bool` nor `Int` is a valid
    GPU kernel scalar arg — see the note on the parameter.)"""
    var b = Int(global_idx.x)
    if b < NS_:
        comptime TM1 = TI_ - 1
        var bootlog = svlog if slowtar != 0 else vlog
        var ret_next = _dev_twp[BINS_](
            bootlog, (b * TI_ + (TI_ - 1)) * BINS_, bins
        )
        var t = TM1 - 1
        while t >= 0:
            var live = rebind[Scalar[DT]](conv[b * TI_ + t + 1])
            var vboot = _dev_twp[BINS_](
                bootlog, (b * TI_ + t + 1) * BINS_, bins
            )
            var interm = (
                rebind[Scalar[DT]](rewv[b * TI_ + t + 1])
                + (Scalar[DT](1.0) - lam) * live * vboot
            )
            var cur = interm + live * lam * ret_next
            ret[b * TM1 + t] = cur
            ret_next = cur
            t -= 1


def _ret_perc_neigh_k[N_: Int](
    ret: Tile[DT, N_],
    neigh: Tile[DT, 4],
    lo_floor_arg: Int64,
    hi_floor_arg: Int64,
):
    """Device-resident percentile *neighbor* finder — replaces the host
    `ret_d.download` + insertion-sort in `PercentileNormalize.update`. Each
    thread `i` computes its STABLE rank in the sorted order (`x[j] < x[i]`, or
    `==` with `j < i` — identical tie-break to `_insertion_sort`), so
    `sorted[rank(i)] == x[i]`. The 4 order-statistics bracketing the lo/hi
    percentiles (`sorted[lo_floor]`, `[lo_floor+1]`, `[hi_floor]`,
    `[hi_floor+1]`) are written by the unique threads whose rank matches — no
    race, no D2H. O(N²) but N = NS·(TI-1) is small and fully parallel. The host
    pre-computes the (constant) floor/frac indices, so this is capture-safe."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var lo_floor = Int(lo_floor_arg)
    var hi_floor = Int(hi_floor_arg)
    var i = Int(global_idx.x)
    if i < N_:
        var xi = rebind[Scalar[DT]](ret[i])
        var rank = 0
        for j in range(N_):
            var xj = rebind[Scalar[DT]](ret[j])
            if xj < xi or (xj == xi and j < i):
                rank += 1
        if rank == lo_floor:
            neigh[0] = xi
        if rank == lo_floor + 1:
            neigh[1] = xi
        if rank == hi_floor:
            neigh[2] = xi
        if rank == hi_floor + 1:
            neigh[3] = xi


def _ret_perc_ema_k(
    neigh: Tile[DT, 4],
    state: Tile[DT, 2],
    rscale: Tile[DT, 1],
    lo_frac: Scalar[DT],
    hi_frac: Scalar[DT],
    rate: Scalar[DT],
    limit: Scalar[DT],
):
    """Single-thread linear-interp percentile + EMA + rscale, all on device —
    the device twin of `PercentileNormalize.update`/`stats` for retnorm's
    `perc`/`debias=False` config. `state=[lo,hi]` is the PERSISTENT EMA (never
    reset); `rscale[0]=max(limit, hi-lo)` is read by `_imag_bwd_k`. Mirrors
    `_percentile_linear` (linear interp) + the `keep·s + rate·p` EMA exactly →
    parity-preserving."""
    if Int(global_idx.x) == 0:
        var plo = rebind[Scalar[DT]](neigh[0]) + lo_frac * (
            rebind[Scalar[DT]](neigh[1]) - rebind[Scalar[DT]](neigh[0])
        )
        var phi = rebind[Scalar[DT]](neigh[2]) + hi_frac * (
            rebind[Scalar[DT]](neigh[3]) - rebind[Scalar[DT]](neigh[2])
        )
        var keep = Scalar[DT](1.0) - rate
        var lo = keep * rebind[Scalar[DT]](state[0]) + rate * plo
        var hi = keep * rebind[Scalar[DT]](state[1]) + rate * phi
        state[0] = lo
        state[1] = hi
        var span = hi - lo
        rscale[0] = limit if limit > span else span


# ── device diagnostic reductions (want_diag log cadence) ───────────────────
# The discrete AC metric readout used to D2H the full imagination histories
# (feats_d 2.2 MB, vlog_d 0.8 MB, pmean_d, …) and sum them on host. Instead
# each metric is reduced on-device (single block, `block.sum`/`block.min` —
# mirrors `DeviceMeanAccum`/`MSELoss`) into a small `[DIAG_N]` buffer, so the
# whole bundle leaves the device in ONE tiny D2H. No accumulation across steps
# (the readout is already gated to log cadence); `want_diag` semantics + the
# parity-gated per-step `last_*_loss` snapshot are preserved exactly.
comptime DIAG_N = 16


def _diag_sum_k[N_: Int](
    data: Tile[DT, N_],
    dst: Tile[DT, DIAG_N],
    slot_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var slot = Int(slot_arg)
    var t = Int(global_idx.x)
    var s: Scalar[DT] = 0.0
    var k = t
    while k < N_:
        s += rebind[Scalar[DT]](data[k])
        k += TPB_REDUCE
    var tot = block.sum[block_size=TPB_REDUCE, broadcast=False](val=s)
    if t == 0:
        dst[slot] = tot[0]


def _diag_sum_sq_k[N_: Int](
    data: Tile[DT, N_],
    dst: Tile[DT, DIAG_N],
    s_sum_arg: Int64,
    s_sq_arg: Int64,
):
    """Σ data → dst[s_sum], Σ data² → dst[s_sq] (one pass; for mean+std)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var s_sum = Int(s_sum_arg)
    var s_sq = Int(s_sq_arg)
    var t = Int(global_idx.x)
    var s: Scalar[DT] = 0.0
    var q: Scalar[DT] = 0.0
    var k = t
    while k < N_:
        var v = rebind[Scalar[DT]](data[k])
        s += v
        q += v * v
        k += TPB_REDUCE
    var ts = block.sum[block_size=TPB_REDUCE, broadcast=False](val=s)
    var tq = block.sum[block_size=TPB_REDUCE, broadcast=False](val=q)
    if t == 0:
        dst[s_sum] = ts[0]
        dst[s_sq] = tq[0]


def _diag_abs_sum_k[N_: Int](
    data: Tile[DT, N_],
    dst: Tile[DT, DIAG_N],
    slot_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var slot = Int(slot_arg)
    var t = Int(global_idx.x)
    var s: Scalar[DT] = 0.0
    var k = t
    while k < N_:
        var v = rebind[Scalar[DT]](data[k])
        s += v if v >= Scalar[DT](0.0) else -v
        k += TPB_REDUCE
    var tot = block.sum[block_size=TPB_REDUCE, broadcast=False](val=s)
    if t == 0:
        dst[slot] = tot[0]


def _diag_min_k[N_: Int](
    data: Tile[DT, N_],
    dst: Tile[DT, DIAG_N],
    slot_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var slot = Int(slot_arg)
    var t = Int(global_idx.x)
    var m: Scalar[DT] = Scalar[DT](1e30)
    var k = t
    while k < N_:
        var v = rebind[Scalar[DT]](data[k])
        if v < m:
            m = v
        k += TPB_REDUCE
    var tot = block.min[block_size=TPB_REDUCE, broadcast=False](val=m)
    if t == 0:
        dst[slot] = tot[0]


def _diag_twohot_sum_sq_k[NBT_: Int, BINS_: Int, NL_: Int, NB_: Int](
    vlog: Tile[DT, NL_],
    bins: Tile[DT, NB_],
    dst: Tile[DT, DIAG_N],
    s_sum_arg: Int64,
    s_sq_arg: Int64,
):
    """Σ twohot_pred(vlog[cell]) and Σ(·)² over the NBT cells → dst[s_sum/s_sq]
    (device twin of the host val_mean/val_std reduction)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var s_sum = Int(s_sum_arg)
    var s_sq = Int(s_sq_arg)
    var t = Int(global_idx.x)
    var s: Scalar[DT] = 0.0
    var q: Scalar[DT] = 0.0
    var c = t
    while c < NBT_:
        var v = _dev_twp[BINS_](vlog, c * BINS_, bins)
        s += v
        q += v * v
        c += TPB_REDUCE
    var ts = block.sum[block_size=TPB_REDUCE, broadcast=False](val=s)
    var tq = block.sum[block_size=TPB_REDUCE, broadcast=False](val=q)
    if t == 0:
        dst[s_sum] = ts[0]
        dst[s_sq] = tq[0]


def _imag_bwd_k[NS_: Int, TI_: Int, BINS_: Int, C_: Int](
    vlog: Tile[DT, NS_ * TI_ * BINS_],
    svlog: Tile[DT, NS_ * TI_ * BINS_],
    pmean: Tile[DT, NS_ * TI_ * C_],
    acts: Tile[DT, NS_ * TI_ * C_],
    conv: Tile[DT, NS_ * TI_],
    ret: Tile[DT, NS_ * (TI_ - 1)],
    bins: Tile[DT, BINS_],
    gvlog: Tile[DT, NS_ * TI_ * BINS_],
    gpmean: Tile[DT, NS_ * TI_ * C_],
    polloss: Tile[DT, NS_ * (TI_ - 1)],
    valloss: Tile[DT, NS_ * (TI_ - 1)],
    rscale_buf: Tile[DT, 1],
    lam: Scalar[DT],
    actent: Scalar[DT],
    slowreg: Scalar[DT],
    inv_im: Scalar[DT],
    u: Scalar[DT],
):
    """Per-start imag-loss forward (loss values) + backward (grads). Computes
    value-logit grads (twohot CE) and policy-logit grads (unimix categorical),
    plus per-(b,t) policy/value losses. Mirrors `imag_loss_cpu`/`_backward`.
    `rscale` is read from a device buffer (written by `_ret_perc_ema_k`) so the
    retnorm scale never round-trips through the host."""
    var b = Int(global_idx.x)
    if b < NS_:
        comptime TM1 = TI_ - 1
        var rscale = rebind[Scalar[DT]](rscale_buf[0])
        # zero this start's grad slices for ALL TI steps (t=TM1.. stay 0).
        for t in range(TI_):
            for c in range(BINS_):
                gvlog[(b * TI_ + t) * BINS_ + c] = Scalar[DT](0.0)
            for a in range(C_):
                gpmean[(b * TI_ + t) * C_ + a] = Scalar[DT](0.0)
        var w_acc = Scalar[DT](1.0)
        for t in range(TM1):
            w_acc *= rebind[Scalar[DT]](conv[b * TI_ + t])  # weight[t]=cumprod
            var w = w_acc
            var vbase = (b * TI_ + t) * BINS_
            var val_t = _dev_twp[BINS_](vlog, vbase, bins)
            var slowval_t = _dev_twp[BINS_](svlog, vbase, bins)
            var ret_t = rebind[Scalar[DT]](ret[b * TM1 + t])
            var adv = (ret_t - val_t) / rscale
            # ── unimix categorical: logp(k) + entropy (k = argmax one-hot) ──
            var pbase = (b * TI_ + t) * C_
            var k = 0
            var bestv = rebind[Scalar[DT]](acts[pbase])
            for a in range(1, C_):
                var av = rebind[Scalar[DT]](acts[pbase + a])
                if av > bestv:
                    bestv = av
                    k = a
            var mx = rebind[Scalar[DT]](pmean[pbase])
            for a in range(1, C_):
                var lv = rebind[Scalar[DT]](pmean[pbase + a])
                if lv > mx:
                    mx = lv
            var sden = Scalar[DT](0.0)
            for a in range(C_):
                sden += exp(rebind[Scalar[DT]](pmean[pbase + a]) - mx)
            var sinv = Scalar[DT](1.0) / sden
            var one_m_u = Scalar[DT](1.0) - u
            var uc = u / Scalar[DT](C_)
            var sm_k = exp(rebind[Scalar[DT]](pmean[pbase + k]) - mx) * sinv
            var p_k = one_m_u * sm_k + uc
            var logpi = log(p_k)
            var ent = Scalar[DT](0.0)
            var ent_dot = Scalar[DT](0.0)
            for m in range(C_):
                var sm_m = exp(rebind[Scalar[DT]](pmean[pbase + m]) - mx) * sinv
                var p_m = one_m_u * sm_m + uc
                ent += -p_m * log(p_m)
                ent_dot += sm_m * (log(p_m) + Scalar[DT](1.0))
            # ── losses ──
            polloss[b * TM1 + t] = w * -(logpi * adv + actent * ent)
            var l1 = _dev_twohot_ce[BINS_](vlog, vbase, bins, ret_t)
            var l2 = _dev_twohot_ce[BINS_](vlog, vbase, bins, slowval_t)
            valloss[b * TM1 + t] = w * (l1 + slowreg * l2)
            # ── policy grads (d_policy cotangent = inv_im) ──
            var dpl_dlogp = inv_im * w * (-adv)
            var dpl_dent = inv_im * w * (-actent)
            var inv_pk = Scalar[DT](1.0) / p_k
            for j in range(C_):
                var sm_j = exp(rebind[Scalar[DT]](pmean[pbase + j]) - mx) * sinv
                var p_j = one_m_u * sm_j + uc
                var delta_kj = Scalar[DT](1.0) if j == k else Scalar[DT](0.0)
                var dlogp = inv_pk * one_m_u * sm_k * (delta_kj - sm_j)
                var dent = -one_m_u * sm_j * (
                    (log(p_j) + Scalar[DT](1.0)) - ent_dot
                )
                gpmean[pbase + j] = dpl_dlogp * dlogp + dpl_dent * dent
            # ── value grads (twohot CE vs ret + vs slowval, d_value=inv_im) ──
            var up = inv_im * w
            _dev_twohot_ce_bwd[BINS_](vlog, vbase, bins, ret_t, up, gvlog)
            _dev_twohot_ce_bwd[BINS_](
                vlog, vbase, bins, slowval_t, up * slowreg, gvlog
            )


def _imag_bwd_continuous_k[NS_: Int, TI_: Int, BINS_: Int, C_: Int](
    vlog: Tile[DT, NS_ * TI_ * BINS_],
    svlog: Tile[DT, NS_ * TI_ * BINS_],
    pmean: Tile[DT, NS_ * TI_ * C_],
    pstd: Tile[DT, NS_ * TI_ * C_],
    acts: Tile[DT, NS_ * TI_ * C_],
    conv: Tile[DT, NS_ * TI_],
    ret: Tile[DT, NS_ * (TI_ - 1)],
    bins: Tile[DT, BINS_],
    gvlog: Tile[DT, NS_ * TI_ * BINS_],
    gpmean: Tile[DT, NS_ * TI_ * C_],
    gpstd: Tile[DT, NS_ * TI_ * C_],
    polloss: Tile[DT, NS_ * (TI_ - 1)],
    valloss: Tile[DT, NS_ * (TI_ - 1)],
    rscale_buf: Tile[DT, 1],
    actent: Scalar[DT],
    slowreg: Scalar[DT],
    inv_im: Scalar[DT],
    minstd: Scalar[DT],
    maxstd: Scalar[DT],
):
    """Continuous (bounded-normal) twin of `_imag_bwd_k`: per-start imag-loss
    forward (polloss/valloss) + backward (gvlog twohot CE, gpmean/gpstd Gaussian
    chain rule). The value path is IDENTICAL to the discrete kernel; only the
    policy distribution differs. Mirrors `imag_loss_cpu` / `imag_loss_backward`
    (DISCRETE=False) bit-for-bit. `ret` + `rscale` precomputed on device."""
    var b = Int(global_idx.x)
    if b < NS_:
        comptime TM1 = TI_ - 1
        comptime LOG2PI = Scalar[DT](1.8378770664093453)
        var rscale = rebind[Scalar[DT]](rscale_buf[0])
        for t in range(TI_):
            for c in range(BINS_):
                gvlog[(b * TI_ + t) * BINS_ + c] = Scalar[DT](0.0)
            for a in range(C_):
                gpmean[(b * TI_ + t) * C_ + a] = Scalar[DT](0.0)
                gpstd[(b * TI_ + t) * C_ + a] = Scalar[DT](0.0)
        var w_acc = Scalar[DT](1.0)
        for t in range(TM1):
            w_acc *= rebind[Scalar[DT]](conv[b * TI_ + t])
            var w = w_acc
            var vbase = (b * TI_ + t) * BINS_
            var val_t = _dev_twp[BINS_](vlog, vbase, bins)
            var slowval_t = _dev_twp[BINS_](svlog, vbase, bins)
            var ret_t = rebind[Scalar[DT]](ret[b * TM1 + t])
            var adv = (ret_t - val_t) / rscale
            var pbase = (b * TI_ + t) * C_
            # ── forward: Σ_a logp / entropy (bounded normal) ──
            var logpi = Scalar[DT](0.0)
            var ent = Scalar[DT](0.0)
            for a in range(C_):
                var mr = rebind[Scalar[DT]](pmean[pbase + a])
                var sr = rebind[Scalar[DT]](pstd[pbase + a])
                var mean = tanh(mr)
                var sg = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-(sr + Scalar[DT](2.0)))
                )
                var std = (maxstd - minstd) * sg + minstd
                var zz = (rebind[Scalar[DT]](acts[pbase + a]) - mean) / std
                logpi += (
                    -Scalar[DT](0.5) * zz * zz - log(std)
                    - Scalar[DT](0.5) * LOG2PI
                )
                ent += Scalar[DT](0.5) * LOG2PI + log(std) + Scalar[DT](0.5)
            polloss[b * TM1 + t] = w * -(logpi * adv + actent * ent)
            var l1 = _dev_twohot_ce[BINS_](vlog, vbase, bins, ret_t)
            var l2 = _dev_twohot_ce[BINS_](vlog, vbase, bins, slowval_t)
            valloss[b * TM1 + t] = w * (l1 + slowreg * l2)
            # ── backward ──
            var dpl_dlogp = inv_im * w * (-adv)
            var dpl_dent = inv_im * w * (-actent)
            for a in range(C_):
                var idx = pbase + a
                var mr = rebind[Scalar[DT]](pmean[idx])
                var sr = rebind[Scalar[DT]](pstd[idx])
                var mean = tanh(mr)
                var sg = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-(sr + Scalar[DT](2.0)))
                )
                var std = (maxstd - minstd) * sg + minstd
                var zz = (rebind[Scalar[DT]](acts[idx]) - mean) / std
                var dlogp_dmean = zz / std
                var dlogp_dstd = (zz * zz - Scalar[DT](1.0)) / std
                var dent_dstd = Scalar[DT](1.0) / std
                var dmean_draw = Scalar[DT](1.0) - mean * mean
                var dstd_draw = (maxstd - minstd) * sg * (Scalar[DT](1.0) - sg)
                gpmean[idx] = dpl_dlogp * dlogp_dmean * dmean_draw
                gpstd[idx] = (
                    dpl_dlogp * dlogp_dstd + dpl_dent * dent_dstd
                ) * dstd_draw
            var up = inv_im * w
            _dev_twohot_ce_bwd[BINS_](vlog, vbase, bins, ret_t, up, gvlog)
            _dev_twohot_ce_bwd[BINS_](
                vlog, vbase, bins, slowval_t, up * slowreg, gvlog
            )


def _imag_bwd_continuous_pt_k[NS_: Int, TI_: Int, BINS_: Int, C_: Int](
    vlog: Tile[DT, NS_ * TI_ * BINS_],
    svlog: Tile[DT, NS_ * TI_ * BINS_],
    pmean: Tile[DT, NS_ * TI_ * C_],
    pstd: Tile[DT, NS_ * TI_ * C_],
    acts: Tile[DT, NS_ * TI_ * C_],
    conv: Tile[DT, NS_ * TI_],
    ret: Tile[DT, NS_ * (TI_ - 1)],
    bins: Tile[DT, BINS_],
    gvlog: Tile[DT, NS_ * TI_ * BINS_],
    gpmean: Tile[DT, NS_ * TI_ * C_],
    gpstd: Tile[DT, NS_ * TI_ * C_],
    polloss: Tile[DT, NS_ * (TI_ - 1)],
    valloss: Tile[DT, NS_ * (TI_ - 1)],
    rscale_buf: Tile[DT, 1],
    actent: Scalar[DT],
    slowreg: Scalar[DT],
    inv_im: Scalar[DT],
    minstd: Scalar[DT],
    maxstd: Scalar[DT],
):
    """Per-(start, t) parallel rewrite of `_imag_bwd_continuous_k`: ONE THREAD per
    (start, imagination step) over NS*(TI-1) — full occupancy + SM spread instead
    of NS=256 threads in one block each running the sequential TM1 scan. The only
    cross-t dependency, the discount cumprod `w`, is recomputed per thread (<=TM1
    mults, negligible vs the BINS twohot work). gvlog/gpmean/gpstd MUST be
    pre-zeroed (gvlog accumulates; the t=TM1 slice stays 0). Math identical to the
    scan kernel (parity-gated)."""
    comptime TM1 = TI_ - 1
    comptime LOG2PI = Scalar[DT](1.8378770664093453)
    var i = Int(global_idx.x)
    if i < NS_ * TM1:
        var b = i // TM1
        var t = i % TM1
        var rscale = rebind[Scalar[DT]](rscale_buf[0])
        # discount cumprod w = prod_{j=0..t} conv[b,j] (the sequential term)
        var w = Scalar[DT](1.0)
        for j in range(t + 1):
            w *= rebind[Scalar[DT]](conv[b * TI_ + j])
        var vbase = (b * TI_ + t) * BINS_
        var val_t = _dev_twp[BINS_](vlog, vbase, bins)
        var slowval_t = _dev_twp[BINS_](svlog, vbase, bins)
        var ret_t = rebind[Scalar[DT]](ret[b * TM1 + t])
        var adv = (ret_t - val_t) / rscale
        var pbase = (b * TI_ + t) * C_
        # ── forward: Σ_a logp / entropy (bounded normal) ──
        var logpi = Scalar[DT](0.0)
        var ent = Scalar[DT](0.0)
        for a in range(C_):
            var mr = rebind[Scalar[DT]](pmean[pbase + a])
            var sr = rebind[Scalar[DT]](pstd[pbase + a])
            var mean = tanh(mr)
            var sg = Scalar[DT](1.0) / (
                Scalar[DT](1.0) + exp(-(sr + Scalar[DT](2.0)))
            )
            var std = (maxstd - minstd) * sg + minstd
            var zz = (rebind[Scalar[DT]](acts[pbase + a]) - mean) / std
            logpi += (
                -Scalar[DT](0.5) * zz * zz - log(std)
                - Scalar[DT](0.5) * LOG2PI
            )
            ent += Scalar[DT](0.5) * LOG2PI + log(std) + Scalar[DT](0.5)
        polloss[b * TM1 + t] = w * -(logpi * adv + actent * ent)
        var l1 = _dev_twohot_ce[BINS_](vlog, vbase, bins, ret_t)
        var l2 = _dev_twohot_ce[BINS_](vlog, vbase, bins, slowval_t)
        valloss[b * TM1 + t] = w * (l1 + slowreg * l2)
        # ── backward ──
        var dpl_dlogp = inv_im * w * (-adv)
        var dpl_dent = inv_im * w * (-actent)
        for a in range(C_):
            var idx = pbase + a
            var mr = rebind[Scalar[DT]](pmean[idx])
            var sr = rebind[Scalar[DT]](pstd[idx])
            var mean = tanh(mr)
            var sg = Scalar[DT](1.0) / (
                Scalar[DT](1.0) + exp(-(sr + Scalar[DT](2.0)))
            )
            var std = (maxstd - minstd) * sg + minstd
            var zz = (rebind[Scalar[DT]](acts[idx]) - mean) / std
            var dlogp_dmean = zz / std
            var dlogp_dstd = (zz * zz - Scalar[DT](1.0)) / std
            var dent_dstd = Scalar[DT](1.0) / std
            var dmean_draw = Scalar[DT](1.0) - mean * mean
            var dstd_draw = (maxstd - minstd) * sg * (Scalar[DT](1.0) - sg)
            gpmean[idx] = dpl_dlogp * dlogp_dmean * dmean_draw
            gpstd[idx] = (
                dpl_dlogp * dlogp_dstd + dpl_dent * dent_dstd
            ) * dstd_draw
        var up = inv_im * w
        _dev_twohot_ce_bwd[BINS_](vlog, vbase, bins, ret_t, up, gvlog)
        _dev_twohot_ce_bwd[BINS_](
            vlog, vbase, bins, slowval_t, up * slowreg, gvlog
        )


def _box_muller_gaussian_k[N_: Int](
    dst: Tile[DT, N_],
    seed: UInt64,
    offset_buf: Tile[DType.uint64, 1],
):
    """Device N(0,1) imagination noise via Box-Muller — the Gaussian counterpart
    of `_noise_philox_k` (which yields uniform[-1,1] for the discrete categorical
    sample). Element i draws two uniforms from Philox(seed+i, offset) and maps
    them z = sqrt(-2·ln u1)·cos(2π·u2). All Scalar[DT] (Metal-safe). The offset
    (advanced by `_noise_off_inc_k`) gives disjoint streams per step. Used only
    when `device_noise=True`; parity runs upload host noise so this is untested
    by the CPU↔GPU gate (stats-checked separately)."""
    var i = Int(global_idx.x)
    if i >= N_:
        return
    var off = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=off)
    var u1 = Scalar[DT](philox.step_uniform()[0])
    var u2 = Scalar[DT](philox.step_uniform()[0])
    if u1 < Scalar[DT](1e-7):
        u1 = Scalar[DT](1e-7)
    var r = sqrt(Scalar[DT](-2.0) * log(u1))
    dst[i] = r * cos(Scalar[DT](6.283185307179586) * u2)


def _repval_setup_k[
    NS_: Int, TI_: Int, FEAT_: Int, B_: Int, T_: Int
](
    ret: Tile[DT, NS_ * (TI_ - 1)],
    feats: Tile[DT, NS_ * TI_ * FEAT_],
    boot: Tile[DT, B_ * T_],
    feat_bt: Tile[DT, B_ * T_ * FEAT_],
):
    """Assemble repval inputs: boot[b,j]=ret[s,0], feat_bt[b,j]=feats[s,0]
    where s=j*B+b (NS=T·B start flattening). Matches `_ac_cpu` repval setup."""
    var i = Int(global_idx.x)
    if i < B_ * T_:
        comptime TM1 = TI_ - 1
        var b = i // T_
        var j = i % T_
        var s = j * B_ + b
        boot[b * T_ + j] = rebind[Scalar[DT]](ret[s * TM1 + 0])
        for k in range(FEAT_):
            feat_bt[(b * T_ + j) * FEAT_ + k] = rebind[Scalar[DT]](
                feats[(s * TI_ + 0) * FEAT_ + k]
            )


def _repl_bwd_k[B_: Int, T_: Int, BINS_: Int](
    last: Tile[DT, B_ * T_],
    rew: Tile[DT, B_ * T_],
    boot: Tile[DT, B_ * T_],
    svlr: Tile[DT, B_ * T_ * BINS_],
    bins: Tile[DT, BINS_],
    gvlr: Tile[DT, B_ * T_ * BINS_],
    vlogits: Tile[DT, B_ * T_ * BINS_],
    horizon: Scalar[DT],
    lam: Scalar[DT],
    slowreg: Scalar[DT],
    inv_rep: Scalar[DT],
):
    """Replay-value loss backward over the REAL replay sequence (term ≡ last
    = mb_dne: the replay stores a single episode-end flag, and the con head
    already trains cont→0 on it, so repval must kill the bootstrap there too
    or the value target disagrees with imagination at episode ends).
    Per-start downward λ-return scan + twohot CE grads into gvlr.
    Mirrors `repl_loss_backward[B,T]`."""
    var b = Int(global_idx.x)
    if b < B_:
        comptime TM1 = T_ - 1
        var disc = Scalar[DT](1.0) - Scalar[DT](1.0) / horizon
        for t in range(T_):
            for c in range(BINS_):
                gvlr[(b * T_ + t) * BINS_ + c] = Scalar[DT](0.0)
        # downward λ-return into ret_next chain; write grads as we know ret[t].
        # need ret[t] for t in 0..TM1-1 → compute full chain first into the
        # bottom-up order using a recompute (T small).
        var ret_next = rebind[Scalar[DT]](boot[b * T_ + (T_ - 1)])
        var t = TM1 - 1
        while t >= 0:
            var live = (
                Scalar[DT](1.0) - rebind[Scalar[DT]](last[b * T_ + t + 1])
            ) * disc
            var cont = (
                Scalar[DT](1.0) - rebind[Scalar[DT]](last[b * T_ + t + 1])
            ) * lam
            var interm = (
                rebind[Scalar[DT]](rew[b * T_ + t + 1])
                + (Scalar[DT](1.0) - cont) * live
                * rebind[Scalar[DT]](boot[b * T_ + t + 1])
            )
            var ret_t = interm + live * cont * ret_next
            ret_next = ret_t
            var w = Scalar[DT](1.0) - rebind[Scalar[DT]](last[b * T_ + t])
            var up = inv_rep * w
            var vbase = (b * T_ + t) * BINS_
            var slowval_t = _dev_twp[BINS_](svlr, vbase, bins)
            _dev_twohot_ce_bwd[BINS_](vlogits, vbase, bins, ret_t, up, gvlr)
            _dev_twohot_ce_bwd[BINS_](
                vlogits, vbase, bins, slowval_t, up * slowreg, gvlr
            )
            t -= 1


# ──────────────────────────────────────────────────────────────────────
# DreamerState — cross-block shared buffers + ctx + inter-block scalars.
# Storage Tensors hold BOTH cpu `.data` and gpu `.dev` (one set per field).
# ──────────────────────────────────────────────────────────────────────


# ── per-action advantage diag (discrete; want_diag-gated). Serial single
#    thread: ~NS·TM1 twohot decodes only at log cadence. out[a] = Σ adv over
#    imagined steps whose SAMPLED action is a; out[C+a] = count. Under
#    REINFORCE E[adv|a] is exactly the per-logit drift direction — a
#    consistent gap between actions while eval is flat = the value/reward
#    heads systematically favor one action's imagined futures (early model
#    exploitation), the policy-collapse driver. ──────────────────────────
def _diag_adv_act_k[NS_: Int, TI_: Int, BINS_: Int, C_: Int](
    acts: Tile[DT, NS_ * TI_ * C_],
    ret: Tile[DT, NS_ * (TI_ - 1)],
    vlog: Tile[DT, NS_ * TI_ * BINS_],
    bins: Tile[DT, BINS_],
    rscale_buf: Tile[DT, 1],
    out_d: Tile[DT, 2 * C_],
):
    if Int(global_idx.x) == 0:
        comptime TM1 = TI_ - 1
        for a in range(2 * C_):
            out_d[a] = Scalar[DT](0.0)
        var rscale = rebind[Scalar[DT]](rscale_buf[0])
        for b in range(NS_):
            for t in range(TM1):
                var pbase = (b * TI_ + t) * C_
                var k = 0
                var bestv = rebind[Scalar[DT]](acts[pbase])
                for a in range(1, C_):
                    var av = rebind[Scalar[DT]](acts[pbase + a])
                    if av > bestv:
                        bestv = av
                        k = a
                var val_t = _dev_twp[BINS_](vlog, (b * TI_ + t) * BINS_, bins)
                var adv = (
                    rebind[Scalar[DT]](ret[b * TM1 + t]) - val_t
                ) / rscale
                out_d[k] = rebind[Scalar[DT]](out_d[k]) + adv
                out_d[C_ + k] = rebind[Scalar[DT]](out_d[C_ + k]) + Scalar[DT](
                    1.0
                )


@fieldwise_init
struct DreamerState[
    OBS: Int, ACT: Int, DETER: Int, SC: Int, TOKEN: Int,
    B: Int, T: Int, T_IMAG: Int,
](Movable & Deinitable):
    var ctx: Optional[DeviceContext]
    # sampled batch (filled by the trainer from replay), batch-major.
    var mb_obs: Tensor   # [B,(T+1),OBS]
    var mb_act: Tensor   # [B,T,ACT]
    var mb_rew: Tensor   # [B,T]
    var mb_dne: Tensor   # [B,T]      transition is_terminal (cont target = 1-dne)
    var mb_fst: Tensor   # [B,T+1]    per-frame is_first (carry-reset mask)
    # RSSM carries (posterior carry sequence).
    var cdeter: Tensor   # [(T+1)*B*DETER]
    var cstoch: Tensor   # [(T+1)*B*SC]
    var toks: Tensor     # [T*B*TOKEN]
    var noise: Tensor    # [T_IMAG*T*B*ACT] (NS=T*B imag starts)
    var last_wm_loss: Scalar[DT]
    var last_ac_loss: Scalar[DT]
    # ── diagnostics (filled per train_step; see docs/runbook) ──
    var dbg_real_rew: Scalar[DT]
    var dbg_rew_pred: Scalar[DT]
    var dbg_ret_mean: Scalar[DT]
    var dbg_ret_std: Scalar[DT]
    var dbg_pmean_abs: Scalar[DT]
    # ── divergence probes ──
    var dbg_val_mean: Scalar[DT]
    var dbg_pstd: Scalar[DT]
    var dbg_rscale: Scalar[DT]
    # ── imagined continue-factor probe (conv = sigmoid(con_logit) over the
    #    imagined rollout). If the cont head never predicts termination these
    #    sit at ~disc (≈0.997) and the λ-return saturates → no actor signal. ──
    var dbg_con_mean: Scalar[DT]
    var dbg_con_min: Scalar[DT]
    # ── collapse probes: spread of value + latent feat over imagined states.
    #    val_std≈0 with healthy feat_std ⇒ value head collapsed to a constant
    #    (no advantage signal); feat_std≈0 ⇒ the latent representation collapsed. ──
    var dbg_val_std: Scalar[DT]
    var dbg_feat_std: Scalar[DT]
    # ── per-component WM loss (per-transition means) + AC split. Surfaced to
    #    the monitoring tool under the World-Model-Losses / KL / Loss groups. ──
    var dbg_dyn_kl: Scalar[DT]
    var dbg_rep_kl: Scalar[DT]
    var dbg_obs_loss: Scalar[DT]
    var dbg_rew_loss: Scalar[DT]
    var dbg_con_loss: Scalar[DT]
    var dbg_pol_loss: Scalar[DT]
    var dbg_val_loss: Scalar[DT]
    # ── per-action mean imagination advantage E[adv | sampled action] (nats-
    #    free, retnorm'd). Discrete only (continuous leaves zeros). A steady
    #    gap between actions while eval is flat = model-exploitation collapse
    #    driver. Length ACT. ──
    var dbg_adv_act: List[Scalar[DT]]
    # ── GPU time-major device minibatch + carries. On CPU these stay empty
    #    (length-0) Tensors. ──
    var d_obs: Tensor    # [T*B*OBS]
    var d_act: Tensor    # [T*B*ACT]
    var d_rew: Tensor    # [T*B]
    var d_cont: Tensor   # [T*B]
    var d_cdeter: Tensor # [(T+1)*B*DETER]
    var d_cstoch: Tensor # [(T+1)*B*SC]
    var d_toks: Tensor   # [T*B*TOKEN]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self(
            ctx=ctx,
            mb_obs=Tensor.make["cpu"](Self.B * (Self.T + 1) * Self.OBS),
            mb_act=Tensor.make["cpu"](Self.B * Self.T * Self.ACT),
            mb_rew=Tensor.make["cpu"](Self.B * Self.T),
            mb_dne=Tensor.make["cpu"](Self.B * Self.T),
            mb_fst=Tensor.make["cpu"](Self.B * (Self.T + 1)),
            cdeter=Tensor.make["cpu"]((Self.T + 1) * Self.B * Self.DETER),
            cstoch=Tensor.make["cpu"]((Self.T + 1) * Self.B * Self.SC),
            toks=Tensor.make["cpu"](Self.T * Self.B * Self.TOKEN),
            noise=Tensor.make["cpu"](Self.T_IMAG * Self.T * Self.B * Self.ACT),
            last_wm_loss=Scalar[DT](0.0),
            last_ac_loss=Scalar[DT](0.0),
            dbg_real_rew=Scalar[DT](0.0),
            dbg_rew_pred=Scalar[DT](0.0),
            dbg_ret_mean=Scalar[DT](0.0),
            dbg_ret_std=Scalar[DT](0.0),
            dbg_pmean_abs=Scalar[DT](0.0),
            dbg_val_mean=Scalar[DT](0.0),
            dbg_pstd=Scalar[DT](0.0),
            dbg_rscale=Scalar[DT](0.0),
            dbg_con_mean=Scalar[DT](0.0),
            dbg_con_min=Scalar[DT](0.0),
            dbg_val_std=Scalar[DT](0.0),
            dbg_feat_std=Scalar[DT](0.0),
            dbg_dyn_kl=Scalar[DT](0.0),
            dbg_rep_kl=Scalar[DT](0.0),
            dbg_obs_loss=Scalar[DT](0.0),
            dbg_rew_loss=Scalar[DT](0.0),
            dbg_con_loss=Scalar[DT](0.0),
            dbg_pol_loss=Scalar[DT](0.0),
            dbg_val_loss=Scalar[DT](0.0),
            dbg_adv_act=List[Scalar[DT]](
                length=Self.ACT, fill=Scalar[DT](0.0)
            ),
            d_obs=Tensor(), d_act=Tensor(), d_rew=Tensor(), d_cont=Tensor(),
            d_cdeter=Tensor(), d_cstoch=Tensor(), d_toks=Tensor(),
        )
        comptime if target == "gpu":
            var c = ctx.value()
            s.d_obs = Tensor.alloc_gpu(c, Self.T * Self.B * Self.OBS)
            s.d_act = Tensor.alloc_gpu(c, Self.T * Self.B * Self.ACT)
            s.d_rew = Tensor.alloc_gpu(c, Self.T * Self.B)
            s.d_cont = Tensor.alloc_gpu(c, Self.T * Self.B)
            s.d_cdeter = Tensor.alloc_gpu(c, (Self.T + 1) * Self.B * Self.DETER)
            s.d_cstoch = Tensor.alloc_gpu(c, (Self.T + 1) * Self.B * Self.SC)
            s.d_toks = Tensor.alloc_gpu(c, Self.T * Self.B * Self.TOKEN)
        return s^


# ──────────────────────────────────────────────────────────────────────
# Device-resident WM-BPTT kernels (Stage 2: move the per-step reset masks /
# carry threading / head-input assembly / loss accumulation off host). The
# carry sequence (cdeter/cstoch), tokens and minibatch live on-device for the
# whole scan; per-step net forwards/vjps queue on the same stream with these
# kernels — ONE sync at the end. Each thread owns a disjoint batch row `b`
# (no cross-thread RMW). Mirrors `_wm_cpu` so the CPU↔GPU parity gate holds.
# ──────────────────────────────────────────────────────────────────────


def _zero_k[N_: Int](buf: Tile[DT, N_]):
    var i = Int(global_idx.x)
    if i < N_:
        buf[i] = Scalar[DT](0.0)


def _wm_slice_store_k[N_: Int, T_: Int](
    src: Tile[DT, N_],
    dst: Tile[DT, T_ * N_],
    t_arg: Int64,
):
    """Store the per-step buffer src[N] into the time-major dst[T,N] at step t."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var i = Int(global_idx.x)
    if i < N_:
        dst[t * N_ + i] = rebind[Scalar[DT]](src[i])


def _wm_carry_in_k[
    B_: Int, D_: Int, SC_: Int, ACT_: Int, TOK_: Int, T_: Int
](
    cdeter: Tile[DT, (T_ + 1) * B_ * D_],
    cstoch: Tile[DT, (T_ + 1) * B_ * SC_],
    mbact: Tile[DT, B_ * T_ * ACT_],
    toks: Tile[DT, T_ * B_ * TOK_],
    mbfst: Tile[DT, B_ * (T_ + 1)],
    cin_d: Tile[DT, B_ * D_],
    cin_s: Tile[DT, B_ * SC_],
    at: Tile[DT, B_ * ACT_],
    tkscr: Tile[DT, B_ * TOK_],
    t_arg: Int64,
):
    """Masked carry/action input + token window for core step t (keep = cut the
    BPTT carry at an episode boundary, from mbfst[b,t+1]). One thread per element
    over B*max(D,SC,TOK) (was B-parallel = a single 16-thread block at B=16, each
    serially looping D+SC+ACT+TOK); each thread writes whichever of cin_d/cin_s/
    at/tkscr its column index k falls within."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    comptime _MDS = D_ if D_ > SC_ else SC_
    comptime MW = _MDS if _MDS > TOK_ else TOK_
    var i = Int(global_idx.x)
    if i < B_ * MW:
        var b = i // MW
        var k = i % MW
        var keep = (
            Scalar[DT](0.0) if rebind[Scalar[DT]](mbfst[b * (T_ + 1) + t + 1]) >= Scalar[DT](0.5)
            else Scalar[DT](1.0)
        )
        if k < D_:
            cin_d[b * D_ + k] = keep * rebind[Scalar[DT]](
                cdeter[t * B_ * D_ + b * D_ + k]
            )
        if k < SC_:
            cin_s[b * SC_ + k] = keep * rebind[Scalar[DT]](
                cstoch[t * B_ * SC_ + b * SC_ + k]
            )
        if k < ACT_:
            at[b * ACT_ + k] = keep * rebind[Scalar[DT]](
                mbact[(b * T_ + t) * ACT_ + k]
            )
        if k < TOK_:
            tkscr[b * TOK_ + k] = rebind[Scalar[DT]](
                toks[t * B_ * TOK_ + b * TOK_ + k]
            )


def _wm_carry_out_k[B_: Int, D_: Int, SC_: Int, CARRY_: Int, T_: Int](
    outbuf: Tile[DT, B_ * CARRY_],
    cdeter: Tile[DT, (T_ + 1) * B_ * D_],
    cstoch: Tile[DT, (T_ + 1) * B_ * SC_],
    klbuf: Tile[DT, T_ * B_ * 2],
    t_arg: Int64,
):
    """Extract next carry (deter/stoch) + the (dyn_kl, rep_kl) losses from the
    core output into the time-major carry sequence + klbuf[t]."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    # one thread per element over B*max(D,SC); the 2 kl scalars on k==0.
    comptime MW = D_ if D_ > SC_ else SC_
    var i = Int(global_idx.x)
    if i < B_ * MW:
        var b = i // MW
        var k = i % MW
        if k < D_:
            cdeter[(t + 1) * B_ * D_ + b * D_ + k] = rebind[Scalar[DT]](
                outbuf[b * CARRY_ + 2 + k]
            )
        if k < SC_:
            cstoch[(t + 1) * B_ * SC_ + b * SC_ + k] = rebind[Scalar[DT]](
                outbuf[b * CARRY_ + 2 + D_ + k]
            )
        if k == 0:
            klbuf[(t * B_ + b) * 2 + 0] = rebind[Scalar[DT]](outbuf[b * CARRY_ + 0])
            klbuf[(t * B_ + b) * 2 + 1] = rebind[Scalar[DT]](outbuf[b * CARRY_ + 1])


def _wm_head_in_k[B_: Int, D_: Int, SC_: Int, OBS_: Int, T_: Int](
    cdeter: Tile[DT, (T_ + 1) * B_ * D_],
    cstoch: Tile[DT, (T_ + 1) * B_ * SC_],
    mbobs: Tile[DT, B_ * (T_ + 1) * OBS_],
    mbrew: Tile[DT, B_ * T_],
    mbdne: Tile[DT, B_ * T_],
    ndn: Tile[DT, B_ * D_],
    snn: Tile[DT, B_ * SC_],
    rtg: Tile[DT, B_ * OBS_],
    rwt: Tile[DT, B_],
    cnt: Tile[DT, B_],
    horizon: Scalar[DT],
    t_arg: Int64,
):
    """Head inputs (next carry) + decoder/reward/continue targets for step t.
    Parallelized over B*OBS (one thread per recon-target pixel) — the dominant
    cost for pixel obs; the small per-b carry (D_/SC_) + reward/continue work is
    done by the FIRST thread of each b's OBS block (k==0), which stays correct
    even when OBS_<D_ (the vector/MLP case). (Was B-parallel with a serial
    OBS-pixel loop → only B threads active.)"""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var i = Int(global_idx.x)
    if i < B_ * OBS_:
        var b = i // OBS_
        var k = i % OBS_
        rtg[i] = rebind[Scalar[DT]](mbobs[(b * (T_ + 1) + t + 1) * OBS_ + k])
        if k == 0:
            var ndbase = (t + 1) * B_ * D_
            var snbase = (t + 1) * B_ * SC_
            for kk in range(D_):
                ndn[b * D_ + kk] = rebind[Scalar[DT]](cdeter[ndbase + b * D_ + kk])
            for kk in range(SC_):
                snn[b * SC_ + kk] = rebind[Scalar[DT]](cstoch[snbase + b * SC_ + kk])
            rwt[b] = rebind[Scalar[DT]](mbrew[b * T_ + t])
            cnt[b] = (Scalar[DT](1.0) - rebind[Scalar[DT]](mbdne[b * T_ + t])) * (
                Scalar[DT](1.0) - Scalar[DT](1.0) / horizon
            )


def _wm_enc_in_k[B_: Int, OBS_: Int, T_: Int](
    mbobs: Tile[DT, B_ * (T_ + 1) * OBS_],
    ob: Tile[DT, B_ * OBS_],
    t_arg: Int64,
):
    """Encoder input window = obs frame t+1 (batch-major). One thread per pixel
    over B*OBS (was B-parallel with a serial OBS loop → only B threads active)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    var i = Int(global_idx.x)
    if i < B_ * OBS_:
        var b = i // OBS_
        var k = i % OBS_
        ob[i] = rebind[Scalar[DT]](mbobs[(b * (T_ + 1) + t + 1) * OBS_ + k])


def _wm_keep_mask_k[B_: Int, D_: Int, SC_: Int, T_: Int](
    gdt: Tile[DT, B_ * D_],
    gst: Tile[DT, B_ * SC_],
    mbfst: Tile[DT, B_ * (T_ + 1)],
    gcd: Tile[DT, B_ * D_],
    gcs: Tile[DT, B_ * SC_],
    t_arg: Int64,
):
    """Cut the BPTT carry gradient at an episode boundary: gcd/gcs = keep·core
    grad_input (keep from mbfst[b,t+1]). One thread per element over B*max(D,SC)
    (was B-parallel = a single 16-thread block at B=16)."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var t = Int(t_arg)
    comptime MW = D_ if D_ > SC_ else SC_
    var i = Int(global_idx.x)
    if i < B_ * MW:
        var b = i // MW
        var k = i % MW
        var keep = (
            Scalar[DT](0.0) if rebind[Scalar[DT]](mbfst[b * (T_ + 1) + t + 1]) >= Scalar[DT](0.5)
            else Scalar[DT](1.0)
        )
        if k < D_:
            gcd[b * D_ + k] = keep * rebind[Scalar[DT]](gdt[b * D_ + k])
        if k < SC_:
            gcs[b * SC_ + k] = keep * rebind[Scalar[DT]](gst[b * SC_ + k])


# ──────────────────────────────────────────────────────────────────────
# WMStep — WM-BPTT over one sampled length-T window. Trains enc/core/dec/
# rew/con; fills state.cdeter / cstoch with the posterior carry sequence.
# ──────────────────────────────────────────────────────────────────────


struct WMStep[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, BINS: Int, B: Int, T: Int,
    DISCRETE: Bool = False,
    # Encoder / decoder Module types (default MLP), threaded from the trainer so
    # WMStep's enc/dec field TYPES match the modules the trainer owns + passes in.
    ENC: Module = DreamerEncoder[OBS, TOKEN, SwishOp],
    DEC: Module = DreamerDecoder[STOCH * CLASSES + DETER, OBS, DEC_U, SwishOp],
    RECON_SIGMOID: Bool = False,  # True → sigmoid+MSE recon (pixel [0,1])
    OUT_INIT: Initializer = Zero,  # reward-head output init (from the trainer)
](Movable & Deinitable):
    # `DISCRETE` selects the WM↔AC carry handoff (Stage 3 P3). The discrete AC
    # (`_ac_gpu_disc`) is fully device-resident, so the GPU WM hands its scan
    # carry straight to the shared `DreamerState` device buffers (no host
    # round-trip). The continuous AC reads the host carry, so its WM keeps the
    # download. Set from `DreamerV3Trainer.DISCRETE`.
    comptime SC = Self.STOCH * Self.CLASSES
    comptime CARRY = 2 + Self.DETER + Self.SC
    comptime EncT = Self.ENC
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN, SwishOp,
    ]
    comptime DecT = DecLossGraph[
        Self.SC, Self.DETER, Self.OBS, Self.DEC_U, SwishOp, Self.DEC,
        Self.RECON_SIGMOID,
    ]
    comptime RewT = RewLossGraph[
        Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp, Self.OUT_INIT
    ]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    comptime StateT = DreamerState[
        Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, 1,
    ]
    var dyn_scale: Scalar[DT]
    var rep_scale: Scalar[DT]
    var horizon: Scalar[DT]   # contdisc continue target = (1-term)·(1-1/horizon)

    # Persistent scratch Tensors (allocated once in make, reused every step).
    var ob: Tensor       # [B*OBS] encoder input window
    var cin_d: Tensor    # [B*DETER] masked carry-deter input
    var cin_s: Tensor    # [B*SC]    masked carry-stoch input
    var at: Tensor       # [B*ACT]   masked prev-action
    var rtg: Tensor      # [B*OBS]   recon target
    var rwt: Tensor      # [B]       reward target
    var cnt: Tensor      # [B]       continue target
    var ndn: Tensor      # [B*DETER] next deter (head input)
    var snn: Tensor      # [B*SC]    next stoch (head input)
    var outbuf: Tensor   # [B*CARRY] core forward output
    var dl: Tensor       # [B]       head-loss readout
    var seed: Tensor     # [B*CARRY] core vjp seed
    var gcd: Tensor      # [B*DETER] carry grad accumulator (deter)
    var gcs: Tensor      # [B*SC]    carry grad accumulator (stoch)
    var ones1: Tensor    # [B]       loss cotangent (1.0)
    var tkscr: Tensor    # [B*TOKEN] enc recompute output
    var gtok: Tensor     # [B*TOKEN] token grad window
    var gobs: Tensor     # [B*OBS]   obs grad (discarded)
    # ── device-resident WM-BPTT buffers (GPU only; empty Tensors on CPU). The
    #    carry sequence / tokens / minibatch stay on-device for the whole scan;
    #    klbuf/obsl/rewl/conl accumulate the per-(t,b) losses for ONE end-of-step
    #    download (last_wm_loss + WM dbg). See `_wm_gpu`. ──
    var mbobs_d: Tensor   # [B*(T+1)*OBS]
    var mbact_d: Tensor   # [B*T*ACT]
    var mbrew_d: Tensor   # [B*T]
    var mbdne_d: Tensor   # [B*T]
    var mbfst_d: Tensor   # [B*(T+1)]
    var cdeter_d: Tensor  # [(T+1)*B*DETER]
    var cstoch_d: Tensor  # [(T+1)*B*SC]
    var toks_d: Tensor    # [T*B*TOKEN]
    var klbuf_d: Tensor   # [T*B*2]  (dyn_kl, rep_kl) per (t,b)
    var obsl_d: Tensor    # [T*B]
    var rewl_d: Tensor    # [T*B]
    var conl_d: Tensor    # [T*B]

    def __init__(out self):
        self.dyn_scale = Scalar[DT](1.0)
        self.rep_scale = Scalar[DT](0.1)
        self.horizon = Scalar[DT](333.0)
        self.ob = Tensor()
        self.cin_d = Tensor()
        self.cin_s = Tensor()
        self.at = Tensor()
        self.rtg = Tensor()
        self.rwt = Tensor()
        self.cnt = Tensor()
        self.ndn = Tensor()
        self.snn = Tensor()
        self.outbuf = Tensor()
        self.dl = Tensor()
        self.seed = Tensor()
        self.gcd = Tensor()
        self.gcs = Tensor()
        self.ones1 = Tensor()
        self.tkscr = Tensor()
        self.gtok = Tensor()
        self.gobs = Tensor()
        self.mbobs_d = Tensor()
        self.mbact_d = Tensor()
        self.mbrew_d = Tensor()
        self.mbdne_d = Tensor()
        self.mbfst_d = Tensor()
        self.cdeter_d = Tensor()
        self.cstoch_d = Tensor()
        self.toks_d = Tensor()
        self.klbuf_d = Tensor()
        self.obsl_d = Tensor()
        self.rewl_d = Tensor()
        self.conl_d = Tensor()

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYl = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime BV = Self.B
        var s = Self()
        # loss_scales dyn=1.0, rep=0.1 — matches the DreamerV3 REFERENCE CODE
        # (configs.yaml global default), which differs from the paper's Eq. 2
        # (dyn=0.5). The reference uses dyn=1.0 for all tasks incl. proprio.
        s.dyn_scale = Scalar[DT](1.0)
        s.rep_scale = Scalar[DT](0.1)
        s.horizon = Scalar[DT](333.0)
        s.ob = _mk[target](BV * OBSD, ctx)
        s.cin_d = _mk[target](BV * D, ctx)
        s.cin_s = _mk[target](BV * SCl, ctx)
        s.at = _mk[target](BV * ACTD, ctx)
        s.rtg = _mk[target](BV * OBSD, ctx)
        s.rwt = _mk[target](BV, ctx)
        s.cnt = _mk[target](BV, ctx)
        s.ndn = _mk[target](BV * D, ctx)
        s.snn = _mk[target](BV * SCl, ctx)
        s.outbuf = _mk[target](BV * CARRYl, ctx)
        s.dl = _mk[target](BV, ctx)
        s.seed = _mk[target](BV * CARRYl, ctx)
        s.gcd = _mk[target](BV * D, ctx)
        s.gcs = _mk[target](BV * SCl, ctx)
        s.ones1 = _mk[target](BV, ctx)
        s.tkscr = _mk[target](BV * TOK, ctx)
        s.gtok = _mk[target](BV * TOK, ctx)
        s.gobs = _mk[target](BV * OBSD, ctx)
        # device-resident WM-BPTT buffers (GPU only).
        comptime Tt = Self.T
        comptime if target == "gpu":
            s.mbobs_d = _mk[target](BV * (Tt + 1) * OBSD, ctx)
            s.mbact_d = _mk[target](BV * Tt * ACTD, ctx)
            s.mbrew_d = _mk[target](BV * Tt, ctx)
            s.mbdne_d = _mk[target](BV * Tt, ctx)
            s.mbfst_d = _mk[target](BV * (Tt + 1), ctx)
            s.cdeter_d = _mk[target]((Tt + 1) * BV * D, ctx)
            s.cstoch_d = _mk[target]((Tt + 1) * BV * SCl, ctx)
            s.toks_d = _mk[target](Tt * BV * TOK, ctx)
            s.klbuf_d = _mk[target](Tt * BV * 2, ctx)
            s.obsl_d = _mk[target](Tt * BV, ctx)
            s.rewl_d = _mk[target](Tt * BV, ctx)
            s.conl_d = _mk[target](Tt * BV, ctx)
            # ones1 is a CONSTANT loss cotangent (all 1.0) — upload it ONCE here
            # so the per-step `_wm_gpu` doesn't H2D-copy it (an upload is illegal
            # inside a CUDA-graph capture; Stage 3 P5).
            for b in range(BV):
                s.ones1.data[b] = Scalar[DT](1.0)
            s.ones1.upload(ctx.value())
        return s^

    def step[
        target: StaticString, T_IMAG: Int,
    ](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, T_IMAG],
        mut enc: Self.EncT,
        mut core: Self.CoreT,
        mut dec: Self.DecT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oe: DreamerOpt,
        mut ocore: DreamerOpt,
        mut odec: DreamerOpt,
        mut orew: DreamerOpt,
        mut ocon: DreamerOpt,
        want_diag: Bool = True,
    ) raises:
        comptime if target == "cpu":
            self._wm_cpu[target, T_IMAG](
                st, enc, core, dec, rew, con, oe, ocore, odec, orew, ocon
            )
        else:
            self._wm_gpu[target, T_IMAG](
                st, enc, core, dec, rew, con, oe, ocore, odec, orew, ocon,
                want_diag,
            )

    def _wm_cpu[
        target: StaticString, T_IMAG: Int,
    ](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, T_IMAG],
        mut enc: Self.EncT,
        mut core: Self.CoreT,
        mut dec: Self.DecT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oe: DreamerOpt,
        mut ocore: DreamerOpt,
        mut odec: DreamerOpt,
        mut orew: DreamerOpt,
        mut ocon: DreamerOpt,
    ) raises:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYl = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        var DYN = self.dyn_scale
        var REP = self.rep_scale
        # Finding 1: observe-step t produces the belief for obs_{t+1} using
        # prev-action a_t — token (+ recon target) = obs frame t+1.
        for t in range(Self.T):
            for b in range(Self.B):
                for k in range(OBSD):
                    self.ob.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
            enc.forward[target, Self.B](
                child_refs[Self.ENC.ARITY, Self.ENC.ACT_DT](self.ob),
                rebind[TensorImpl[Self.ENC.ACT_DT]](self.tkscr),
                None,
            )
            var base = t * Self.B * TOK
            for i in range(Self.B * TOK):
                st.toks.data[base + i] = self.tkscr.data[i]
        for i in range(Self.B * D):
            st.cdeter.data[i] = 0.0
        for i in range(Self.B * SCl):
            st.cstoch.data[i] = 0.0
        var total: Scalar[DT] = 0.0
        # per-component accumulators (per-transition means → metrics)
        var acc_dyn: Scalar[DT] = 0.0
        var acc_rep: Scalar[DT] = 0.0
        var acc_obs: Scalar[DT] = 0.0
        var acc_rew: Scalar[DT] = 0.0
        var acc_con: Scalar[DT] = 0.0
        # forward scan
        for t in range(Self.T):
            var dbase = t * Self.B * D
            var sbase = t * Self.B * SCl
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.cin_d.data[b * D + k] = keep * st.cdeter.data[dbase + b * D + k]
                for k in range(SCl):
                    self.cin_s.data[b * SCl + k] = keep * st.cstoch.data[sbase + b * SCl + k]
                for k in range(ACTD):
                    self.at.data[b * ACTD + k] = keep * st.mb_act.data[(b * Self.T + t) * ACTD + k]
            # tokens window for step t into a fresh Tensor for set_input
            for i in range(Self.B * TOK):
                self.tkscr.data[i] = st.toks.data[t * Self.B * TOK + i]
            core.set_input["deter", Self.B](self.cin_d, None)
            core.set_input["stoch", Self.B](self.cin_s, None)
            core.set_input["action", Self.B](self.at, None)
            core.set_input["tokens", Self.B](self.tkscr, None)
            core.forward[Self.B, target](self.outbuf, None)
            var ndbase = (t + 1) * Self.B * D
            var snbase = (t + 1) * Self.B * SCl
            for b in range(Self.B):
                for k in range(D):
                    st.cdeter.data[ndbase + b * D + k] = self.outbuf.data[b * CARRYl + 2 + k]
                for k in range(SCl):
                    st.cstoch.data[snbase + b * SCl + k] = self.outbuf.data[b * CARRYl + 2 + D + k]
                total += DYN * self.outbuf.data[b * CARRYl + 0] + REP * self.outbuf.data[b * CARRYl + 1]
                acc_dyn += self.outbuf.data[b * CARRYl + 0]
                acc_rep += self.outbuf.data[b * CARRYl + 1]
            # head inputs (next carry) + targets
            for b in range(Self.B):
                for k in range(D):
                    self.ndn.data[b * D + k] = st.cdeter.data[ndbase + b * D + k]
                for k in range(SCl):
                    self.snn.data[b * SCl + k] = st.cstoch.data[snbase + b * SCl + k]
                for k in range(OBSD):
                    self.rtg.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
                self.rwt.data[b] = st.mb_rew.data[b * Self.T + t]
                self.cnt.data[b] = (Scalar[DT](1.0) - st.mb_dne.data[b * Self.T + t]) * (
                    Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
                )
            dec.set_input["stoch_new", Self.B](self.snn, None)
            dec.set_input["nd", Self.B](self.ndn, None)
            dec.set_input["rtgt", Self.B](self.rtg, None)
            dec.forward[Self.B, target](self.dl, None)
            for b in range(Self.B):
                total += self.dl.data[b]
                acc_obs += self.dl.data[b]
            rew.set_input["nd", Self.B](self.ndn, None)
            rew.set_input["stoch_new", Self.B](self.snn, None)
            rew.set_input["rtgt", Self.B](self.rwt, None)
            rew.forward[Self.B, target](self.dl, None)
            for b in range(Self.B):
                total += self.dl.data[b]
                acc_rew += self.dl.data[b]
            con.set_input["nd", Self.B](self.ndn, None)
            con.set_input["stoch_new", Self.B](self.snn, None)
            con.set_input["ctgt", Self.B](self.cnt, None)
            con.forward[Self.B, target](self.dl, None)
            for b in range(Self.B):
                total += self.dl.data[b]
                acc_con += self.dl.data[b]

        # zero grads. enc is a Module → opt.zero_grad over the module; the loss
        # graphs OWN their params (a ComputeGraph is not a Module) → zero them via
        # the graph's own zero_grad.
        oe.zero_grad[target, M=Self.EncT](enc, None)
        core.zero_grad[target](None)
        dec.zero_grad[target](None)
        rew.zero_grad[target](None)
        con.zero_grad[target](None)
        for i in range(Self.B * D):
            self.gcd.data[i] = 0.0
        for i in range(Self.B * SCl):
            self.gcs.data[i] = 0.0
        for b in range(Self.B):
            self.ones1.data[b] = 1.0
        # backward scan
        for rev in range(Self.T):
            var t = Self.T - 1 - rev
            var dbase = t * Self.B * D
            var sbase = t * Self.B * SCl
            var ndbase = (t + 1) * Self.B * D
            var snbase = (t + 1) * Self.B * SCl
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.cin_d.data[b * D + k] = keep * st.cdeter.data[dbase + b * D + k]
                for k in range(SCl):
                    self.cin_s.data[b * SCl + k] = keep * st.cstoch.data[sbase + b * SCl + k]
                for k in range(ACTD):
                    self.at.data[b * ACTD + k] = keep * st.mb_act.data[(b * Self.T + t) * ACTD + k]
                for k in range(D):
                    self.ndn.data[b * D + k] = st.cdeter.data[ndbase + b * D + k]
                for k in range(SCl):
                    self.snn.data[b * SCl + k] = st.cstoch.data[snbase + b * SCl + k]
                for k in range(OBSD):
                    self.rtg.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
                self.rwt.data[b] = st.mb_rew.data[b * Self.T + t]
                self.cnt.data[b] = (Scalar[DT](1.0) - st.mb_dne.data[b * Self.T + t]) * (
                    Scalar[DT](1.0) - Scalar[DT](1.0) / self.horizon
                )
            # dec
            dec.set_input["stoch_new", Self.B](self.snn, None)
            dec.set_input["nd", Self.B](self.ndn, None)
            dec.set_input["rtgt", Self.B](self.rtg, None)
            dec.forward[Self.B, target](self.dl, None)
            dec.vjp[Self.B, target](self.ones1, None)
            # rew
            rew.set_input["nd", Self.B](self.ndn, None)
            rew.set_input["stoch_new", Self.B](self.snn, None)
            rew.set_input["rtgt", Self.B](self.rwt, None)
            rew.forward[Self.B, target](self.dl, None)
            rew.vjp[Self.B, target](self.ones1, None)
            # con
            con.set_input["nd", Self.B](self.ndn, None)
            con.set_input["stoch_new", Self.B](self.snn, None)
            con.set_input["ctgt", Self.B](self.cnt, None)
            con.forward[Self.B, target](self.dl, None)
            con.vjp[Self.B, target](self.ones1, None)
            # assemble the core vjp seed
            ref dnd = dec.grad_input["nd"]()
            ref dsn = dec.grad_input["stoch_new"]()
            ref rnd = rew.grad_input["nd"]()
            ref rsn = rew.grad_input["stoch_new"]()
            ref cnd = con.grad_input["nd"]()
            ref csn = con.grad_input["stoch_new"]()
            for b in range(Self.B):
                self.seed.data[b * CARRYl + 0] = DYN
                self.seed.data[b * CARRYl + 1] = REP
                for k in range(D):
                    self.seed.data[b * CARRYl + 2 + k] = (
                        self.gcd.data[b * D + k] + dnd.data[b * D + k]
                        + rnd.data[b * D + k] + cnd.data[b * D + k]
                    )
                for k in range(SCl):
                    self.seed.data[b * CARRYl + 2 + D + k] = (
                        self.gcs.data[b * SCl + k] + dsn.data[b * SCl + k]
                        + rsn.data[b * SCl + k] + csn.data[b * SCl + k]
                    )
            # tokens window for step t
            for i in range(Self.B * TOK):
                self.tkscr.data[i] = st.toks.data[t * Self.B * TOK + i]
            core.set_input["deter", Self.B](self.cin_d, None)
            core.set_input["stoch", Self.B](self.cin_s, None)
            core.set_input["action", Self.B](self.at, None)
            core.set_input["tokens", Self.B](self.tkscr, None)
            core.forward[Self.B, target](self.outbuf, None)
            core.vjp[Self.B, target](self.seed, None)
            ref gdt = core.grad_input["deter"]()
            ref gst = core.grad_input["stoch"]()
            # Finding 3: cut the BPTT carry gradient at an episode boundary.
            for b in range(Self.B):
                var keep = (
                    Scalar[DT](0.0) if st.mb_fst.data[b * (Self.T + 1) + t + 1] >= Scalar[DT](0.5)
                    else Scalar[DT](1.0)
                )
                for k in range(D):
                    self.gcd.data[b * D + k] = keep * gdt.data[b * D + k]
                for k in range(SCl):
                    self.gcs.data[b * SCl + k] = keep * gst.data[b * SCl + k]
            # encoder backward — re-encode obs frame t+1, vjp the token grad.
            ref gtok = core.grad_input["tokens"]()
            for i in range(Self.B * TOK):
                self.gtok.data[i] = gtok.data[i]
            for b in range(Self.B):
                for k in range(OBSD):
                    self.ob.data[b * OBSD + k] = st.mb_obs.data[
                        (b * (Self.T + 1) + t + 1) * OBSD + k
                    ]
            enc.forward[target, Self.B](
                child_refs[Self.ENC.ARITY, Self.ENC.ACT_DT](self.ob),
                rebind[TensorImpl[Self.ENC.ACT_DT]](self.tkscr),
                None,
            )
            enc.vjp[target, Self.B](
                child_refs[Self.ENC.ARITY, Self.ENC.ACT_DT](self.ob),
                rebind[TensorImpl[Self.ENC.ACT_DT]](self.gtok),
                child_refs[Self.ENC.ARITY, Self.ENC.ACT_DT](self.gobs),
                None,
            )
        # optimizer steps
        oe.step[target, M=Self.EncT](enc, None)
        ocore.begin_step()
        walk_params[target](core, ocore, None)
        odec.begin_step()
        walk_params[target](dec, odec, None)
        orew.begin_step()
        walk_params[target](rew, orew, None)
        ocon.begin_step()
        walk_params[target](con, ocon, None)
        var _nbt = Scalar[DT](Self.B * Self.T)
        st.dbg_dyn_kl = acc_dyn / _nbt
        st.dbg_rep_kl = acc_rep / _nbt
        st.dbg_obs_loss = acc_obs / _nbt
        st.dbg_rew_loss = acc_rew / _nbt
        st.dbg_con_loss = acc_con / _nbt
        st.last_wm_loss = total

    def _wm_gpu[
        target: StaticString, T_IMAG: Int,
    ](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, T_IMAG],
        mut enc: Self.EncT,
        mut core: Self.CoreT,
        mut dec: Self.DecT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oe: DreamerOpt,
        mut ocore: DreamerOpt,
        mut odec: DreamerOpt,
        mut orew: DreamerOpt,
        mut ocon: DreamerOpt,
        want_diag: Bool = True,
    ) raises:
        # GPU WM-BPTT scan — device-resident (Stage 2). The carry sequence
        # (cdeter_d/cstoch_d), tokens (toks_d) and minibatch live on-device for
        # the whole scan; per-step reset masks / carry threading / head-input
        # assembly run as kernels and the head losses accumulate into device
        # buffers (klbuf/obsl/rewl/conl) for ONE end-of-step download. Net
        # forwards/vjps + kernels queue on the ctx stream → a single sync at the
        # end. Mirrors `_wm_cpu` so the CPU↔GPU parity gate holds.
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYl = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime Tt = Self.T
        comptime CD = (Tt + 1) * Self.B * D
        comptime CS = (Tt + 1) * Self.B * SCl
        comptime TKD = Tt * Self.B * TOK
        var DYN = self.dyn_scale
        var REP = self.rep_scale
        var ctx = st.ctx.value()
        comptime nbB = (Self.B + TPB - 1) // TPB
        comptime nbBO = (Self.B * OBSD + TPB - 1) // TPB  # one thread / pixel
        comptime _MDS = D if D > SCl else SCl
        comptime _MW = _MDS if _MDS > TOK else TOK
        comptime nbBW = (Self.B * _MW + TPB - 1) // TPB  # B*max(D,SC,TOK)
        comptime nbTOK = (Self.B * TOK + TPB - 1) // TPB

        # The batch-major device minibatch (mbobs_d/mbact_d/mbrew_d/mbdne_d/
        # mbfst_d) is now filled directly by the trainer — `GPUSequenceReplay.
        # sample_batch_fst_dev` samples straight into these buffers (Stage 3
        # P2b), so the per-step host gather + H2D upload that used to live here
        # is gone. The sample kernels queue on `ctx` ahead of this scan → the
        # data is ready by the time the WM reads it (same-stream ordering).

        # ── encode tokens: enc(obs frame t+1) → toks_d[t] (device-resident) ──
        for t in range(Self.T):
            ctx.enqueue_function[_wm_enc_in_k[Self.B, OBSD, Tt]](
                self.mbobs_d.tt["gpu", Self.B * (Tt + 1) * OBSD](),
                self.ob.tt["gpu", Self.B * OBSD](),
                Int64(t), grid_dim=nbBO, block_dim=TPB,
            )
            enc.forward[target, Self.B](
                child_refs[Self.ENC.ARITY, Self.ENC.ACT_DT](self.ob),
                rebind[TensorImpl[Self.ENC.ACT_DT]](self.tkscr),
                ctx,
            )
            ctx.enqueue_function[_wm_slice_store_k[Self.B * TOK, Tt]](
                self.tkscr.tt["gpu", Self.B * TOK](),
                self.toks_d.tt["gpu", TKD](),
                Int64(t), grid_dim=nbTOK, block_dim=TPB,
            )

        # zero the carry sequence (init carry at index 0 must be 0).
        ctx.enqueue_function[_zero_k[CD]](
            self.cdeter_d.tt["gpu", CD](),
            grid_dim=(CD + TPB - 1) // TPB, block_dim=TPB,
        )
        ctx.enqueue_function[_zero_k[CS]](
            self.cstoch_d.tt["gpu", CS](),
            grid_dim=(CS + TPB - 1) // TPB, block_dim=TPB,
        )

        # ── forward scan (no per-step host sync) ──
        for t in range(Self.T):
            ctx.enqueue_function[_wm_carry_in_k[Self.B, D, SCl, ACTD, TOK, Tt]](
                self.cdeter_d.tt["gpu", CD](),
                self.cstoch_d.tt["gpu", CS](),
                self.mbact_d.tt["gpu", Self.B * Tt * ACTD](),
                self.toks_d.tt["gpu", TKD](),
                self.mbfst_d.tt["gpu", Self.B * (Tt + 1)](),
                self.cin_d.tt["gpu", Self.B * D](),
                self.cin_s.tt["gpu", Self.B * SCl](),
                self.at.tt["gpu", Self.B * ACTD](),
                self.tkscr.tt["gpu", Self.B * TOK](),
                Int64(t), grid_dim=nbBW, block_dim=TPB,
            )
            core.set_input["deter", Self.B](self.cin_d, ctx)
            core.set_input["stoch", Self.B](self.cin_s, ctx)
            core.set_input["action", Self.B](self.at, ctx)
            core.set_input["tokens", Self.B](self.tkscr, ctx)
            core.forward[Self.B, target](self.outbuf, ctx)
            ctx.enqueue_function[_wm_carry_out_k[Self.B, D, SCl, CARRYl, Tt]](
                self.outbuf.tt["gpu", Self.B * CARRYl](),
                self.cdeter_d.tt["gpu", CD](),
                self.cstoch_d.tt["gpu", CS](),
                self.klbuf_d.tt["gpu", Tt * Self.B * 2](),
                Int64(t), grid_dim=nbBW, block_dim=TPB,
            )
            ctx.enqueue_function[_wm_head_in_k[Self.B, D, SCl, OBSD, Tt]](
                self.cdeter_d.tt["gpu", CD](),
                self.cstoch_d.tt["gpu", CS](),
                self.mbobs_d.tt["gpu", Self.B * (Tt + 1) * OBSD](),
                self.mbrew_d.tt["gpu", Self.B * Tt](),
                self.mbdne_d.tt["gpu", Self.B * Tt](),
                self.ndn.tt["gpu", Self.B * D](),
                self.snn.tt["gpu", Self.B * SCl](),
                self.rtg.tt["gpu", Self.B * OBSD](),
                self.rwt.tt["gpu", Self.B](),
                self.cnt.tt["gpu", Self.B](),
                self.horizon, Int64(t), grid_dim=nbBO, block_dim=TPB,
            )
            dec.set_input["stoch_new", Self.B](self.snn, ctx)
            dec.set_input["nd", Self.B](self.ndn, ctx)
            dec.set_input["rtgt", Self.B](self.rtg, ctx)
            dec.forward[Self.B, target](self.dl, ctx)
            ctx.enqueue_function[_wm_slice_store_k[Self.B, Tt]](
                self.dl.tt["gpu", Self.B](),
                self.obsl_d.tt["gpu", Tt * Self.B](),
                Int64(t), grid_dim=nbB, block_dim=TPB,
            )
            rew.set_input["nd", Self.B](self.ndn, ctx)
            rew.set_input["stoch_new", Self.B](self.snn, ctx)
            rew.set_input["rtgt", Self.B](self.rwt, ctx)
            rew.forward[Self.B, target](self.dl, ctx)
            ctx.enqueue_function[_wm_slice_store_k[Self.B, Tt]](
                self.dl.tt["gpu", Self.B](),
                self.rewl_d.tt["gpu", Tt * Self.B](),
                Int64(t), grid_dim=nbB, block_dim=TPB,
            )
            con.set_input["nd", Self.B](self.ndn, ctx)
            con.set_input["stoch_new", Self.B](self.snn, ctx)
            con.set_input["ctgt", Self.B](self.cnt, ctx)
            con.forward[Self.B, target](self.dl, ctx)
            ctx.enqueue_function[_wm_slice_store_k[Self.B, Tt]](
                self.dl.tt["gpu", Self.B](),
                self.conl_d.tt["gpu", Tt * Self.B](),
                Int64(t), grid_dim=nbB, block_dim=TPB,
            )

        # zero grads (enc Module via opt; loss graphs own their params) + the
        # carry-grad accumulators; ones1 = the per-head loss cotangent.
        oe.zero_grad[target, M=Self.EncT](enc, ctx)
        core.zero_grad[target](ctx)
        dec.zero_grad[target](ctx)
        rew.zero_grad[target](ctx)
        con.zero_grad[target](ctx)
        ctx.enqueue_function[_zero_k[Self.B * D]](
            self.gcd.tt["gpu", Self.B * D](),
            grid_dim=(Self.B * D + TPB - 1) // TPB, block_dim=TPB,
        )
        ctx.enqueue_function[_zero_k[Self.B * SCl]](
            self.gcs.tt["gpu", Self.B * SCl](),
            grid_dim=(Self.B * SCl + TPB - 1) // TPB, block_dim=TPB,
        )
        # ones1 (constant loss cotangent) is uploaded ONCE in make — no per-step
        # H2D here (illegal inside a CUDA-graph capture; Stage 3 P5).
        # ── backward scan (no per-step host sync) ──
        for rev in range(Self.T):
            var t = Self.T - 1 - rev
            # masked carry/action/tokens input + head inputs/targets (device).
            ctx.enqueue_function[_wm_carry_in_k[Self.B, D, SCl, ACTD, TOK, Tt]](
                self.cdeter_d.tt["gpu", CD](),
                self.cstoch_d.tt["gpu", CS](),
                self.mbact_d.tt["gpu", Self.B * Tt * ACTD](),
                self.toks_d.tt["gpu", TKD](),
                self.mbfst_d.tt["gpu", Self.B * (Tt + 1)](),
                self.cin_d.tt["gpu", Self.B * D](),
                self.cin_s.tt["gpu", Self.B * SCl](),
                self.at.tt["gpu", Self.B * ACTD](),
                self.tkscr.tt["gpu", Self.B * TOK](),
                Int64(t), grid_dim=nbBW, block_dim=TPB,
            )
            ctx.enqueue_function[_wm_head_in_k[Self.B, D, SCl, OBSD, Tt]](
                self.cdeter_d.tt["gpu", CD](),
                self.cstoch_d.tt["gpu", CS](),
                self.mbobs_d.tt["gpu", Self.B * (Tt + 1) * OBSD](),
                self.mbrew_d.tt["gpu", Self.B * Tt](),
                self.mbdne_d.tt["gpu", Self.B * Tt](),
                self.ndn.tt["gpu", Self.B * D](),
                self.snn.tt["gpu", Self.B * SCl](),
                self.rtg.tt["gpu", Self.B * OBSD](),
                self.rwt.tt["gpu", Self.B](),
                self.cnt.tt["gpu", Self.B](),
                self.horizon, Int64(t), grid_dim=nbBO, block_dim=TPB,
            )
            # dec
            dec.set_input["stoch_new", Self.B](self.snn, ctx)
            dec.set_input["nd", Self.B](self.ndn, ctx)
            dec.set_input["rtgt", Self.B](self.rtg, ctx)
            dec.forward[Self.B, target](self.dl, ctx)
            dec.vjp[Self.B, target](self.ones1, ctx)
            # rew
            rew.set_input["nd", Self.B](self.ndn, ctx)
            rew.set_input["stoch_new", Self.B](self.snn, ctx)
            rew.set_input["rtgt", Self.B](self.rwt, ctx)
            rew.forward[Self.B, target](self.dl, ctx)
            rew.vjp[Self.B, target](self.ones1, ctx)
            # con
            con.set_input["nd", Self.B](self.ndn, ctx)
            con.set_input["stoch_new", Self.B](self.snn, ctx)
            con.set_input["ctgt", Self.B](self.cnt, ctx)
            con.forward[Self.B, target](self.dl, ctx)
            con.vjp[Self.B, target](self.ones1, ctx)
            # assemble the core vjp seed on device via _seed_asm_k (reads the
            # head grad_inputs + carry-grad accumulators gcd/gcs, all device).
            ctx.enqueue_function[_seed_asm_k[Self.B, CARRYl, D, SCl]](
                self.seed.tt["gpu", Self.B * CARRYl](),
                self.gcd.tt["gpu", Self.B * D](),
                self.gcs.tt["gpu", Self.B * SCl](),
                dec.grad_input["nd"]().tt["gpu", Self.B * D](),
                rew.grad_input["nd"]().tt["gpu", Self.B * D](),
                con.grad_input["nd"]().tt["gpu", Self.B * D](),
                dec.grad_input["stoch_new"]().tt["gpu", Self.B * SCl](),
                rew.grad_input["stoch_new"]().tt["gpu", Self.B * SCl](),
                con.grad_input["stoch_new"]().tt["gpu", Self.B * SCl](),
                DYN, REP, grid_dim=nbBW, block_dim=TPB,
            )
            core.set_input["deter", Self.B](self.cin_d, ctx)
            core.set_input["stoch", Self.B](self.cin_s, ctx)
            core.set_input["action", Self.B](self.at, ctx)
            core.set_input["tokens", Self.B](self.tkscr, ctx)
            core.forward[Self.B, target](self.outbuf, ctx)
            core.vjp[Self.B, target](self.seed, ctx)
            # Finding 3: cut the BPTT carry gradient at an episode boundary —
            # row-scale the core grad_inputs by the keep mask into gcd/gcs (device).
            ctx.enqueue_function[_wm_keep_mask_k[Self.B, D, SCl, Tt]](
                core.grad_input["deter"]().tt["gpu", Self.B * D](),
                core.grad_input["stoch"]().tt["gpu", Self.B * SCl](),
                self.mbfst_d.tt["gpu", Self.B * (Tt + 1)](),
                self.gcd.tt["gpu", Self.B * D](),
                self.gcs.tt["gpu", Self.B * SCl](),
                Int64(t), grid_dim=nbBW, block_dim=TPB,
            )
            # encoder backward — re-encode obs frame t+1, vjp the token grad.
            ref gtok = core.grad_input["tokens"]()
            ctx.enqueue_copy(self.gtok.dev.value(), gtok.dev.value())
            ctx.enqueue_function[_wm_enc_in_k[Self.B, OBSD, Tt]](
                self.mbobs_d.tt["gpu", Self.B * (Tt + 1) * OBSD](),
                self.ob.tt["gpu", Self.B * OBSD](),
                Int64(t), grid_dim=nbBO, block_dim=TPB,
            )
            enc.forward[target, Self.B](
                child_refs[Self.ENC.ARITY, Self.ENC.ACT_DT](self.ob),
                rebind[TensorImpl[Self.ENC.ACT_DT]](self.tkscr),
                ctx,
            )
            enc.vjp[target, Self.B](
                child_refs[Self.ENC.ARITY, Self.ENC.ACT_DT](self.ob),
                rebind[TensorImpl[Self.ENC.ACT_DT]](self.gtok),
                child_refs[Self.ENC.ARITY, Self.ENC.ACT_DT](self.gobs),
                ctx,
            )
        # optimizer steps (device bias-correction advance — capture-safe)
        oe.step[target, M=Self.EncT](enc, ctx)
        ocore.begin_step_gpu(ctx)
        walk_params[target](core, ocore, ctx)
        odec.begin_step_gpu(ctx)
        walk_params[target](dec, odec, ctx)
        orew.begin_step_gpu(ctx)
        walk_params[target](rew, orew, ctx)
        ocon.begin_step_gpu(ctx)
        walk_params[target](con, ocon, ctx)
        # ── end-of-step readout (Stage 3 P3). ──
        # Carry handoff: copy the scan carry straight into the shared DreamerState
        # device buffers the device-resident AC reads — for BOTH discrete
        # (`_ac_gpu_disc`) and continuous (`_ac_gpu_cont`); no D2H/H2D round-trip,
        # async → the carry handoff is CUDA-graph capturable for both. (st.d_cdeter
        # ← cdeter_d is exactly what the continuous AC used to read from host
        # st.cdeter, which was itself a download of cdeter_d — so this is
        # equivalent, just without the host round-trip.)
        ctx.enqueue_copy(st.d_cdeter.dev.value(), self.cdeter_d.dev.value())
        ctx.enqueue_copy(st.d_cstoch.dev.value(), self.cstoch_d.dev.value())
        # WM loss + dbg readout is purely diagnostic (the optimizer steps above
        # already applied the gradients) → gate it to `want_diag` (P3b). On the
        # non-diag path this skips the ONLY D2H + the `synchronize`, leaving the WM
        # step fully async (CUDA-graph capturable) — for both discrete + continuous.
        var need_sync = want_diag
        if want_diag:
            self.klbuf_d.download(ctx)
            self.obsl_d.download(ctx)
            self.rewl_d.download(ctx)
            self.conl_d.download(ctx)
        if need_sync:
            ctx.synchronize()
        if want_diag:
            var total: Scalar[DT] = 0.0
            var acc_dyn: Scalar[DT] = 0.0
            var acc_rep: Scalar[DT] = 0.0
            var acc_obs: Scalar[DT] = 0.0
            var acc_rew: Scalar[DT] = 0.0
            var acc_con: Scalar[DT] = 0.0
            for i in range(Self.T * Self.B):
                var dk = self.klbuf_d.data[i * 2 + 0]
                var rk = self.klbuf_d.data[i * 2 + 1]
                var ol = self.obsl_d.data[i]
                var rl = self.rewl_d.data[i]
                var cl = self.conl_d.data[i]
                acc_dyn += dk
                acc_rep += rk
                acc_obs += ol
                acc_rew += rl
                acc_con += cl
                total += DYN * dk + REP * rk + ol + rl + cl
            var _nbt = Scalar[DT](Self.B * Self.T)
            st.dbg_dyn_kl = acc_dyn / _nbt
            st.dbg_rep_kl = acc_rep / _nbt
            st.dbg_obs_loss = acc_obs / _nbt
            st.dbg_rew_loss = acc_rew / _nbt
            st.dbg_con_loss = acc_con / _nbt
            st.last_wm_loss = total


# ──────────────────────────────────────────────────────────────────────
# ParamSyncStep — copy core/prior params WMCoreGraph → WMImagineGraph.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct ParamSyncStep[
    DETER: Int, H: Int, STOCH: Int, CLASSES: Int, BLOCKS: Int, ACT: Int,
    TOKEN: Int,
](Movable & Deinitable):
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN, SwishOp,
    ]
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def step[target: StaticString](
        mut self, mut core: Self.CoreT, mut imagine: Self.ImagT,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # GPU: device-direct mirror (no host round-trip → capture-safe; Stage 3
        # P5). CPU: the host snapshot/apply (a round-trip is fine eagerly).
        comptime if target == "gpu":
            var snap = collect_graph_params_device[target](core, ctx)
            apply_graph_params_device[target](imagine, snap^, ctx.value())
        else:
            var snap = collect_graph_params[target](core, ctx)
            apply_graph_params[target](imagine, snap, ctx=ctx)


# ──────────────────────────────────────────────────────────────────────
# ACStep — imagination rollout + actor-critic loss. Trains value/policy;
# Polyak-updates slowvalue. Reads the start carry from state.cdeter[T].
# ──────────────────────────────────────────────────────────────────────


struct ACStep[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, HU: Int, VU: Int, PU: Int, BINS: Int,
    B: Int, T: Int, T_IMAG: Int, DISCRETE: Bool = False,
    OUT_INIT: Initializer = Zero,  # reward/value output init (from the trainer)
](Movable & Deinitable):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]
    comptime ValT = DreamerValue[
        Self.FEAT, Self.VU, Self.BINS, SwishOp, Self.OUT_INIT
    ]
    comptime PolT = DreamerPolicyHead[
        Self.FEAT, Self.PU, Self.ACT, Self.DISCRETE, SwishOp
    ]
    comptime POUT = Self.ACT if Self.DISCRETE else 2 * Self.ACT
    comptime RewT = RewLossGraph[
        Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp, Self.OUT_INIT
    ]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    var minstd: Scalar[DT]
    var maxstd: Scalar[DT]
    var lam: Scalar[DT]
    var actent: Scalar[DT]
    var slowreg: Scalar[DT]
    var slow_rate: Scalar[DT]
    var horizon: Scalar[DT]
    var repval_scale: Scalar[DT]
    var slowtar: Bool

    # Persistent scratch Tensors (NS = T*B imagination starts; BT = B*T == NS).
    var fb: Tensor       # [NS*FEAT] per-step feat
    var pb: Tensor       # [NS*POUT] policy output
    var vb: Tensor       # [NS*BINS] value output
    var svb: Tensor      # [NS*BINS] slowvalue output
    var cd: Tensor       # [NS*DETER] rollout carry deter
    var cs: Tensor       # [NS*SC]    rollout carry stoch
    var at: Tensor       # [NS*ACT]   imagined action
    var dummy1: Tensor   # [NS]       head-loss dummy
    var vscr: Tensor     # [NS*BINS]  value re-forward scratch
    var pscr: Tensor     # [NS*POUT]  policy re-forward scratch
    var ftt: Tensor      # [NS*FEAT]  per-step feat (backward)
    var gvt: Tensor      # [NS*BINS]  value grad-out window
    var gfeat: Tensor    # [NS*FEAT]  value/policy grad-in (discarded)
    var polg: Tensor     # [NS*POUT]  policy grad-out window
    var feat_bt: Tensor  # [BT*FEAT]  repval features
    var vlr: Tensor      # [BT*BINS]  repval value forward
    var svlr: Tensor     # [BT*BINS]  repval slowvalue forward
    var g_vlr: Tensor    # [BT*BINS]  repval value grad-out
    var grf: Tensor      # [BT*FEAT]  repval value grad-in (discarded)
    # ── device-resident discrete-AC histories [NS, TI, W] + loss scratch.
    #    Allocated GPU-only (empty Tensors on CPU). Keep the imagination
    #    rollout + λ-return + imag/repl loss on-device (no per-step host
    #    marshalling); see `_ac_gpu_disc`. ──
    var feats_d: Tensor  # [NS*TI*FEAT]
    var pmean_d: Tensor  # [NS*TI*ACT]  policy logits (discrete)
    var vlog_d: Tensor   # [NS*TI*BINS]
    var svlog_d: Tensor  # [NS*TI*BINS]
    var acts_d: Tensor   # [NS*TI*ACT]  one-hot sampled action
    var rewv_d: Tensor   # [NS*TI]
    var conv_d: Tensor   # [NS*TI]
    var ret_d: Tensor    # [NS*TM1]     λ-return (downloaded for retnorm)
    var gvlog_d: Tensor  # [NS*TI*BINS] value-logit grads
    var gpmean_d: Tensor # [NS*TI*ACT]  policy mean-raw grads (logits, discrete)
    var pstd_d: Tensor   # [NS*TI*ACT]  policy std-raw history (CONTINUOUS only)
    var gpstd_d: Tensor  # [NS*TI*ACT]  policy std-raw grads   (CONTINUOUS only)
    var polloss_d: Tensor # [NS*TM1]
    var valloss_d: Tensor # [NS*TM1]
    var boot_d: Tensor   # [BT]         repval bootstrap (= ret[s,0])
    var mbdne_d: Tensor  # [BT]         mb_dne on device (repval `last`)
    var mbrew_d: Tensor  # [BT]         mb_rew on device
    var noise_d: Tensor  # [TI*NS*ACT]  imagination noise on device
    # Device-resident Philox offset for the production noise generator (P4
    # follow-up). `None` until allocated (GPU+DISCRETE); advanced per step so
    # each step draws a fresh stream. `noise_seed` is the fixed base seed.
    var noise_off_d: Optional[DeviceBuffer[DType.uint64]]
    var noise_seed: UInt64
    var bins_d: Tensor   # [BINS]       twohot grid on device
    var retstate_d: Tensor  # [2]  PERSISTENT retnorm EMA [lo, hi] (never reset)
    var neigh_d: Tensor  # [4]  per-step percentile neighbors (scratch)
    var rscale_d: Tensor # [1]  retnorm scale (read by _imag_bwd_k on device)
    var diag_d: Tensor   # [DIAG_N]  device-reduced metric bundle (1 D2H/flush)
    var adv_diag_d: Tensor  # [2*ACT]  per-action adv sums+counts (diag-only)

    def __init__(out self):
        self.minstd = Scalar[DT](0.1)
        self.maxstd = Scalar[DT](1.0)
        self.lam = Scalar[DT](0.95)
        self.actent = Scalar[DT](3e-4)
        self.slowreg = Scalar[DT](1.0)
        self.slow_rate = Scalar[DT](0.02)
        self.horizon = Scalar[DT](333.0)
        self.repval_scale = Scalar[DT](0.3)
        self.slowtar = False
        self.fb = Tensor()
        self.pb = Tensor()
        self.vb = Tensor()
        self.svb = Tensor()
        self.cd = Tensor()
        self.cs = Tensor()
        self.at = Tensor()
        self.dummy1 = Tensor()
        self.vscr = Tensor()
        self.pscr = Tensor()
        self.ftt = Tensor()
        self.gvt = Tensor()
        self.gfeat = Tensor()
        self.polg = Tensor()
        self.feat_bt = Tensor()
        self.vlr = Tensor()
        self.svlr = Tensor()
        self.g_vlr = Tensor()
        self.grf = Tensor()
        self.feats_d = Tensor()
        self.pmean_d = Tensor()
        self.vlog_d = Tensor()
        self.svlog_d = Tensor()
        self.acts_d = Tensor()
        self.rewv_d = Tensor()
        self.conv_d = Tensor()
        self.ret_d = Tensor()
        self.gvlog_d = Tensor()
        self.gpmean_d = Tensor()
        self.pstd_d = Tensor()
        self.gpstd_d = Tensor()
        self.polloss_d = Tensor()
        self.valloss_d = Tensor()
        self.boot_d = Tensor()
        self.mbdne_d = Tensor()
        self.mbrew_d = Tensor()
        self.noise_d = Tensor()
        self.noise_off_d = None
        self.noise_seed = UInt64(0x5EED_D17E_A12C_0FFE)
        self.bins_d = Tensor()
        self.retstate_d = Tensor()
        self.neigh_d = Tensor()
        self.rscale_d = Tensor()
        self.diag_d = Tensor()
        self.adv_diag_d = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        actent: Scalar[DT] = Scalar[DT](3e-4),
        slowtar: Bool = False,
    ) raises -> Self:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime POUTl = Self.POUT
        comptime NS = Self.T * Self.B
        comptime BT = Self.B * Self.T
        var s = Self()
        s.actent = actent
        s.slowtar = slowtar
        s.fb = _mk[target](NS * FEATl, ctx)
        s.pb = _mk[target](NS * POUTl, ctx)
        s.vb = _mk[target](NS * BINSl, ctx)
        s.svb = _mk[target](NS * BINSl, ctx)
        s.cd = _mk[target](NS * D, ctx)
        s.cs = _mk[target](NS * SCl, ctx)
        s.at = _mk[target](NS * ACTD, ctx)
        s.dummy1 = _mk[target](NS, ctx)
        s.vscr = _mk[target](NS * BINSl, ctx)
        s.pscr = _mk[target](NS * POUTl, ctx)
        s.ftt = _mk[target](NS * FEATl, ctx)
        s.gvt = _mk[target](NS * BINSl, ctx)
        s.gfeat = _mk[target](NS * FEATl, ctx)
        s.polg = _mk[target](NS * POUTl, ctx)
        s.feat_bt = _mk[target](BT * FEATl, ctx)
        s.vlr = _mk[target](BT * BINSl, ctx)
        s.svlr = _mk[target](BT * BINSl, ctx)
        s.g_vlr = _mk[target](BT * BINSl, ctx)
        s.grf = _mk[target](BT * FEATl, ctx)
        # device-resident AC histories (GPU only). Allocated for BOTH discrete
        # (`_ac_gpu_disc`) and continuous (`_ac_gpu_cont`) — pstd_d / gpstd_d are
        # continuous-only (the bounded-normal std-raw history + its grad).
        comptime TM1 = Self.T_IMAG - 1
        comptime TIl = Self.T_IMAG
        comptime if target == "gpu":
            s.feats_d = _mk[target](NS * TIl * FEATl, ctx)
            s.pmean_d = _mk[target](NS * TIl * ACTD, ctx)
            s.pstd_d = _mk[target](NS * TIl * ACTD, ctx)
            s.vlog_d = _mk[target](NS * TIl * BINSl, ctx)
            s.svlog_d = _mk[target](NS * TIl * BINSl, ctx)
            s.acts_d = _mk[target](NS * TIl * ACTD, ctx)
            s.rewv_d = _mk[target](NS * TIl, ctx)
            s.conv_d = _mk[target](NS * TIl, ctx)
            s.ret_d = _mk[target](NS * TM1, ctx)
            s.gvlog_d = _mk[target](NS * TIl * BINSl, ctx)
            s.gpmean_d = _mk[target](NS * TIl * ACTD, ctx)
            s.gpstd_d = _mk[target](NS * TIl * ACTD, ctx)
            s.polloss_d = _mk[target](NS * TM1, ctx)
            s.valloss_d = _mk[target](NS * TM1, ctx)
            s.boot_d = _mk[target](BT, ctx)
            s.mbdne_d = _mk[target](BT, ctx)
            s.mbrew_d = _mk[target](BT, ctx)
            s.noise_d = _mk[target](TIl * NS * ACTD, ctx)
            # device Philox offset for production noise (P4 follow-up): [1]
            # uint64, 0. A stable buffer (never recreated) so it's capture-safe.
            var noff = ctx.value().enqueue_create_buffer[DType.uint64](1)
            noff.enqueue_fill(UInt64(0))
            s.noise_off_d = noff^
            s.bins_d = _mk[target](BINSl, ctx)
            # device-resident retnorm: persistent EMA state zeroed once (matches
            # PercentileNormalize.__init__ lo=hi=0); neigh/rscale are per-step.
            s.retstate_d = _mk[target](2, ctx)
            s.retstate_d.data[0] = Scalar[DT](0.0)
            s.retstate_d.data[1] = Scalar[DT](0.0)
            s.retstate_d.upload(ctx.value())
            s.neigh_d = _mk[target](4, ctx)
            s.rscale_d = _mk[target](1, ctx)
            s.diag_d = _mk[target](DIAG_N, ctx)
            s.adv_diag_d = _mk[target](2 * ACTD, ctx)
        return s^

    def gen_noise_device(mut self, ctx: DeviceContext) raises:
        """Generate the imagination noise on-device via Philox into `noise_d`
        and advance the offset (Stage 3 P4 follow-up; GPU+DISCRETE only).
        Replaces the host `random_float64` gen + H2D upload — pure device work.
        Capture-safe: `noise_d` is a fixed buffer (never recreated) the captured
        AC reads, and the offset is device-resident; run from the eager prologue
        each step so successive steps draw disjoint streams."""
        comptime NSC = Self.T_IMAG * Self.T * Self.B * Self.ACT
        comptime nbN = (NSC + TPB - 1) // TPB
        var off_lt = tile[1](self.noise_off_d.value())
        ctx.enqueue_function[_noise_philox_k[NSC]](
            self.noise_d.tt["gpu", NSC](),
            self.noise_seed, off_lt,
            grid_dim=nbN, block_dim=TPB,
        )
        ctx.enqueue_function[_noise_off_inc_k[NSC]](
            off_lt, grid_dim=1, block_dim=1,
        )

    def step[target: StaticString](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, Self.T_IMAG],
        mut imagine: Self.ImagT,
        mut value: Self.ValT,
        mut slowvalue: Self.ValT,
        mut policy: Self.PolT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oval: DreamerOpt,
        mut opol: DreamerOpt,
        mut retnorm: PercentileNormalize,
        bins: Pointer[Scalar[DT], MutAnyOrigin],
        want_diag: Bool = True,
    ) raises:
        # `want_diag` gates the host diagnostic/loss readouts on the
        # device-resident GPU path (the only consumer); the training math runs
        # regardless. CPU + continuous GPU ignore it (always compute).
        comptime if target == "cpu":
            self._ac_cpu[target](
                st, imagine, value, slowvalue, policy, rew, con,
                oval, opol, retnorm, bins,
            )
        else:
            comptime if Self.DISCRETE:
                self._ac_gpu_disc[target](
                    st, imagine, value, slowvalue, policy, rew, con,
                    oval, opol, retnorm, bins, want_diag,
                )
            else:
                self._ac_gpu_cont[target](
                    st, imagine, value, slowvalue, policy, rew, con,
                    oval, opol, retnorm, bins, want_diag,
                )

    def _ac_cpu[target: StaticString](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, Self.T_IMAG],
        mut imagine: Self.ImagT,
        mut value: Self.ValT,
        mut slowvalue: Self.ValT,
        mut policy: Self.PolT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oval: DreamerOpt,
        mut opol: DreamerOpt,
        mut retnorm: PercentileNormalize,
        bins: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime TI = Self.T_IMAG
        comptime POUTl = Self.POUT
        var MINSTD = self.minstd
        var MAXSTD = self.maxstd
        # FIX (2): imagine from ALL B·T posterior carries (cdeter indices 1..T;
        # index 0 is the zero init) flattened to NS = T·B starts.
        comptime NS = Self.T * Self.B

        # host accumulator arrays (List → RAII, no leaks) for the loss helpers.
        var acts = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var pmean = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var pstd = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var vlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var svlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var rewv = List[Scalar[DT]](length=NS * TI, fill=Scalar[DT](0))
        var conv = List[Scalar[DT]](length=NS * TI, fill=Scalar[DT](0))
        var feats = List[Scalar[DT]](length=NS * TI * FEATl, fill=Scalar[DT](0))

        # init rollout carry cd/cs from posterior carries 1..T.
        for i in range(NS * D):
            self.cd.data[i] = st.cdeter.data[Self.B * D + i]
        for i in range(NS * SCl):
            self.cs.data[i] = st.cstoch.data[Self.B * SCl + i]

        for t in range(TI):
            for b in range(NS):
                for k in range(D):
                    self.fb.data[b * FEATl + k] = self.cd.data[b * D + k]
                for k in range(SCl):
                    self.fb.data[b * FEATl + D + k] = self.cs.data[b * SCl + k]
                for k in range(FEATl):
                    feats[(b * TI + t) * FEATl + k] = self.fb.data[b * FEATl + k]
            policy.forward[target, NS](TensorRefs[1](self.fb), self.pb, None)
            comptime if Self.DISCRETE:
                for b in range(NS):
                    for a in range(ACTD):
                        pmean[(b * TI + t) * ACTD + a] = self.pb.data[b * ACTD + a]
                        pstd[(b * TI + t) * ACTD + a] = 0.0
                    var z0 = st.noise.data[(t * NS + b) * ACTD + 0]
                    var u01 = (z0 + Scalar[DT](1.0)) * Scalar[DT](0.5)
                    var k = cat_sample[ACTD](_hp(self.pb), b * ACTD, UNIMIX, u01)
                    for a in range(ACTD):
                        acts[(b * TI + t) * ACTD + a] = (
                            Scalar[DT](1.0) if a == k else Scalar[DT](0.0)
                        )
            else:
                for b in range(NS):
                    for a in range(ACTD):
                        var mr = self.pb.data[b * 2 * ACTD + a]
                        var sr = self.pb.data[b * 2 * ACTD + ACTD + a]
                        pmean[(b * TI + t) * ACTD + a] = mr
                        pstd[(b * TI + t) * ACTD + a] = sr
                        var z = st.noise.data[(t * NS + b) * ACTD + a]
                        acts[(b * TI + t) * ACTD + a] = (
                            tanh(mr) + bounded_std(sr, MINSTD, MAXSTD) * z
                        )
            value.forward[target, NS](TensorRefs[1](self.fb), self.vb, None)
            slowvalue.forward[target, NS](TensorRefs[1](self.fb), self.svb, None)
            for b in range(NS):
                for c in range(BINSl):
                    vlog[(b * TI + t) * BINSl + c] = self.vb.data[b * BINSl + c]
                    svlog[(b * TI + t) * BINSl + c] = self.svb.data[b * BINSl + c]
            # rew head — read its logits from node_output["rew"].
            rew.set_input["nd", NS](self.cd, None)
            rew.set_input["stoch_new", NS](self.cs, None)
            rew.set_input["rtgt", NS](self.dummy1, None)
            rew.forward[NS, target](self.dummy1, None)
            ref rew_logits = rew.node_output["rew"]()
            con.set_input["nd", NS](self.cd, None)
            con.set_input["stoch_new", NS](self.cs, None)
            con.set_input["ctgt", NS](self.dummy1, None)
            con.forward[NS, target](self.dummy1, None)
            ref con_logit = con.node_output["con"]()
            for b in range(NS):
                rewv[b * TI + t] = twohot_pred[BINSl](
                    _hp(rew_logits), b * BINSl, bins
                )
                conv[b * TI + t] = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-con_logit.data[b])
                )
                for a in range(ACTD):
                    self.at.data[b * ACTD + a] = acts[(b * TI + t) * ACTD + a]
            imagine.set_input["deter", NS](self.cd, None)
            imagine.set_input["stoch", NS](self.cs, None)
            imagine.set_input["action", NS](self.at, None)
            imagine.forward[NS, target](self.fb, None)
            ref nd = imagine.node_output["nd"]()
            ref sn = imagine.node_output["stoch_new"]()
            for i in range(NS * D):
                self.cd.data[i] = nd.data[i]
            for i in range(NS * SCl):
                self.cs.data[i] = sn.data[i]

        comptime TM1 = TI - 1
        var pol_loss = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var val_loss = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var ret = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        imag_loss_cpu[NS, TI, ACTD, BINSl, Self.DISCRETE](
            acts.unsafe_ptr(), rewv.unsafe_ptr(), conv.unsafe_ptr(),
            vlog.unsafe_ptr(), svlog.unsafe_ptr(), pmean.unsafe_ptr(),
            pstd.unsafe_ptr(), bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            retnorm, pol_loss, val_loss,
            ret, self.slowtar,
        )
        var total: Scalar[DT] = 0.0
        var pol_sum: Scalar[DT] = 0.0
        var val_sum: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            total += pol_loss[i] + val_loss[i]
            pol_sum += pol_loss[i]
            val_sum += val_loss[i]
        var _inv_ac = Scalar[DT](1.0) / Scalar[DT](NS * TM1)
        total = total * _inv_ac
        st.dbg_pol_loss = pol_sum * _inv_ac
        st.dbg_val_loss = val_sum * _inv_ac
        # ── diagnostics ──
        var pma: Scalar[DT] = 0.0
        for i in range(NS * TI * ACTD):
            pma += pmean[i] if pmean[i] >= 0 else -pmean[i]
        st.dbg_pmean_abs = pma / Scalar[DT](NS * TI * ACTD)
        var rp: Scalar[DT] = 0.0
        for i in range(NS * TI):
            rp += rewv[i]
        st.dbg_rew_pred = rp / Scalar[DT](NS * TI)
        # imagined continue-factor (conv): mean + min over the rollout. min≈0.997
        # ⇒ the cont head never truncates → λ-return saturated → no actor signal.
        var cm: Scalar[DT] = 0.0
        var cmin: Scalar[DT] = conv[0]
        for i in range(NS * TI):
            cm += conv[i]
            if conv[i] < cmin:
                cmin = conv[i]
        st.dbg_con_mean = cm / Scalar[DT](NS * TI)
        st.dbg_con_min = cmin
        var rm: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            rm += ret[i]
        rm = rm / Scalar[DT](NS * TM1)
        st.dbg_ret_mean = rm
        var rv: Scalar[DT] = 0.0
        for i in range(NS * TM1):
            var dd = ret[i] - rm
            rv += dd * dd
        st.dbg_ret_std = sqrt(rv / Scalar[DT](NS * TM1))
        var rscale = retnorm.stats()[1]
        st.dbg_rscale = rscale
        # per-action mean advantage E[adv | sampled action] (discrete only) —
        # the REINFORCE per-logit drift direction; mirrors _diag_adv_act_k.
        comptime if Self.DISCRETE:
            var adv_s = List[Scalar[DT]](length=ACTD, fill=Scalar[DT](0))
            var adv_n = List[Scalar[DT]](length=ACTD, fill=Scalar[DT](0))
            for b in range(NS):
                for t in range(TM1):
                    var pbase = (b * TI + t) * ACTD
                    var k = 0
                    var bestv = acts[pbase]
                    for a in range(1, ACTD):
                        if acts[pbase + a] > bestv:
                            bestv = acts[pbase + a]
                            k = a
                    var val_t = twohot_pred[BINSl](
                        vlog.unsafe_ptr(), (b * TI + t) * BINSl, bins
                    )
                    var adv = (ret[b * TM1 + t] - val_t) / rscale
                    adv_s[k] += adv
                    adv_n[k] += Scalar[DT](1.0)
            for a in range(ACTD):
                st.dbg_adv_act[a] = (
                    adv_s[a] / adv_n[a]
                    if adv_n[a] > Scalar[DT](0.0) else Scalar[DT](0.0)
                )
        var ps_acc: Scalar[DT] = 0.0
        comptime if not Self.DISCRETE:
            for i in range(NS * TI * ACTD):
                ps_acc += bounded_std(pstd[i], MINSTD, MAXSTD)
        st.dbg_pstd = ps_acc / Scalar[DT](NS * TI * ACTD)
        var vm_acc: Scalar[DT] = 0.0
        for b in range(NS):
            for t in range(TI):
                vm_acc += twohot_pred[BINSl](vlog.unsafe_ptr(), (b * TI + t) * BINSl, bins)
        var vmean = vm_acc / Scalar[DT](NS * TI)
        st.dbg_val_mean = vmean
        # value spread over imagined states (val_std≈0 ⇒ value head collapsed)
        var vv: Scalar[DT] = 0.0
        for b in range(NS):
            for t in range(TI):
                var dv = twohot_pred[BINSl](vlog.unsafe_ptr(), (b * TI + t) * BINSl, bins) - vmean
                vv += dv * dv
        st.dbg_val_std = sqrt(vv / Scalar[DT](NS * TI))
        # latent feat spread over imagined states (feat_std≈0 ⇒ latent collapsed)
        var fm: Scalar[DT] = 0.0
        for i in range(NS * TI * FEATl):
            fm += feats[i]
        fm = fm / Scalar[DT](NS * TI * FEATl)
        var fv: Scalar[DT] = 0.0
        for i in range(NS * TI * FEATl):
            var df = feats[i] - fm
            fv += df * df
        st.dbg_feat_std = sqrt(fv / Scalar[DT](NS * TI * FEATl))

        var d_pol = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var d_val = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
        var inv_im = Scalar[DT](1.0) / Scalar[DT](NS * TM1)
        for i in range(NS * TM1):
            d_pol[i] = inv_im
            d_val[i] = inv_im
        var g_vlog = List[Scalar[DT]](length=NS * TI * BINSl, fill=Scalar[DT](0))
        var g_pmean = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        var g_pstd = List[Scalar[DT]](length=NS * TI * ACTD, fill=Scalar[DT](0))
        imag_loss_backward[NS, TI, ACTD, BINSl, Self.DISCRETE](
            acts.unsafe_ptr(), rewv.unsafe_ptr(), conv.unsafe_ptr(),
            vlog.unsafe_ptr(), svlog.unsafe_ptr(), pmean.unsafe_ptr(),
            pstd.unsafe_ptr(), bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            rscale, d_pol.unsafe_ptr(), d_val.unsafe_ptr(),
            g_vlog.unsafe_ptr(), g_pmean.unsafe_ptr(), g_pstd.unsafe_ptr(),
            self.slowtar,
        )
        oval.zero_grad[target, M=Self.ValT](value, None)
        opol.zero_grad[target, M=Self.PolT](policy, None)
        for t in range(TI):
            for b in range(NS):
                for k in range(FEATl):
                    self.ftt.data[b * FEATl + k] = feats[(b * TI + t) * FEATl + k]
            value.forward[target, NS](TensorRefs[1](self.ftt), self.vscr, None)
            for b in range(NS):
                for c in range(BINSl):
                    self.gvt.data[b * BINSl + c] = g_vlog[(b * TI + t) * BINSl + c]
            value.vjp[target, NS](
                TensorRefs[1](self.ftt), self.gvt, TensorRefs[1](self.gfeat), None
            )
            policy.forward[target, NS](TensorRefs[1](self.ftt), self.pscr, None)
            comptime if Self.DISCRETE:
                for b in range(NS):
                    for a in range(ACTD):
                        self.polg.data[b * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
            else:
                for b in range(NS):
                    for a in range(ACTD):
                        self.polg.data[b * 2 * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
                        self.polg.data[b * 2 * ACTD + ACTD + a] = g_pstd[(b * TI + t) * ACTD + a]
            policy.vjp[target, NS](
                TensorRefs[1](self.ftt), self.polg, TensorRefs[1](self.gfeat), None
            )

        # ── repval: ground the value head on REAL replay transitions ──
        comptime BT = Self.B * Self.T
        var boot_bt = List[Scalar[DT]](length=BT, fill=Scalar[DT](0))
        var term_bt = List[Scalar[DT]](length=BT, fill=Scalar[DT](0))
        for b in range(Self.B):
            for j in range(Self.T):
                var s = j * Self.B + b
                boot_bt[b * Self.T + j] = ret[s * TM1 + 0]
                # term = dne: the replay's one episode-end flag. The con head
                # already trains cont→0 on it, so repval must not bootstrap
                # through it (was hardcoded 0 → ret = rew + disc·boot at real
                # terminals instead of rew).
                term_bt[b * Self.T + j] = st.mb_dne.data[b * Self.T + j]
                for k in range(FEATl):
                    self.feat_bt.data[(b * Self.T + j) * FEATl + k] = (
                        feats[(s * TI) * FEATl + k]
                    )
        value.forward[target, BT](TensorRefs[1](self.feat_bt), self.vlr, None)
        slowvalue.forward[target, BT](TensorRefs[1](self.feat_bt), self.svlr, None)
        # repval runs over the REAL replay sequence (length Self.T), NOT the
        # imagination horizon: repl_loss_backward[Self.B, Self.T] emits a
        # cotangent shaped [Self.B, Self.T-1]. Size d_rep accordingly (TM1 above
        # is the imagination TM1 = T_IMAG-1; reusing it overflows when T_IMAG≠T).
        comptime TM1R = Self.T - 1
        var d_rep = List[Scalar[DT]](length=Self.B * TM1R, fill=Scalar[DT](0))
        var inv_rep = self.repval_scale / Scalar[DT](Self.B * TM1R)
        for i in range(Self.B * TM1R):
            d_rep[i] = inv_rep
        repl_loss_backward[Self.B, Self.T, BINSl](
            _hp(st.mb_dne), term_bt, _hp(st.mb_rew),
            boot_bt, _hp(self.vlr), _hp(self.svlr), bins,
            self.horizon, self.lam, self.slowreg, d_rep,
            _hp(self.g_vlr),
        )
        value.vjp[target, BT](
            TensorRefs[1](self.feat_bt), self.g_vlr, TensorRefs[1](self.grf), None
        )

        oval.step[target, M=Self.ValT](value, None)
        opol.step[target, M=Self.PolT](policy, None)
        polyak_module[target, Self.ValT](value, slowvalue, self.slow_rate)
        st.last_ac_loss = total

    # (the superseded host-round-trip `_ac_gpu` was deleted 2026-07: zero
    # callers — continuous GPU uses `_ac_gpu_cont`, discrete `_ac_gpu_disc`
    # — and it carried a live under-launch bug, grid sized NS where its
    # kernel needed NS*FEAT threads. Recover from git history if needed.)

    def _ac_gpu_disc[target: StaticString](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, Self.T_IMAG],
        mut imagine: Self.ImagT,
        mut value: Self.ValT,
        mut slowvalue: Self.ValT,
        mut policy: Self.PolT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oval: DreamerOpt,
        mut opol: DreamerOpt,
        mut retnorm: PercentileNormalize,
        bins: Pointer[Scalar[DT], MutAnyOrigin],
        want_diag: Bool,
    ) raises:
        # Device-resident discrete imagination-AC. The whole rollout + λ-return
        # + imag/repl loss runs on-device through the kernels above (histories
        # [NS,TI,W]); the ONLY host round-trips are: a one-time noise/bins/mb
        # upload, ONE `ret` download for the percentile retnorm (a 2-scalar EMA),
        # and the gated diagnostic readout. Training is identical to `_ac_cpu`
        # (parity-gated); `want_diag` only toggles the host metric scalars.
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime TI = Self.T_IMAG
        comptime NS = Self.T * Self.B
        comptime TM1 = TI - 1
        comptime BT = Self.B * Self.T
        comptime TM1R = Self.T - 1
        comptime nbB = (NS + TPB - 1) // TPB
        comptime nbNF = (NS * FEATl + TPB - 1) // TPB  # one thread / hist element
        comptime BLKS = 32  # one warp = 32 starts
        comptime nbBS = (NS + BLKS - 1) // BLKS  # spread NS starts over SMs
        comptime nbBT = (BT + TPB - 1) // TPB
        comptime nbB1 = (Self.B + TPB - 1) // TPB
        var ctx = st.ctx.value()

        # Imagination noise + the constant bins grid are BOTH already on device,
        # uploaded OUT of this (captured) region: noise in the trainer prologue
        # (P4), bins once in DreamerV3Trainer.make (P5). An H2D upload is illegal
        # inside a CUDA-graph capture, so neither can be staged here.
        # rew/dne come device-direct from the shared state buffers (Stage 3
        # P3c): `_draw_minibatch`/`load_minibatch` left the sampled [B*T] reward
        # window in st.d_rew and the done window in st.d_cont, so copy them
        # straight into the AC buffers — no host round-trip (capture-safe).
        ctx.enqueue_copy(self.mbrew_d.dev.value(), st.d_rew.dev.value())
        ctx.enqueue_copy(self.mbdne_d.dev.value(), st.d_cont.dev.value())

        # init rollout carry cd/cs from posterior carries 1..T — device-direct
        # from the WM (Stage 3 P3): `_wm_gpu` left the scan carry in the shared
        # state device buffers, so copy the [1..T] slice (skip the index-0 init
        # carry: offset B*D / B*SC, length NS*D / NS*SC) straight into cd/cs.
        var sub_cd = st.d_cdeter.dev.value().create_sub_buffer[DT](
            Self.B * D, NS * D
        )
        var sub_cs = st.d_cstoch.dev.value().create_sub_buffer[DT](
            Self.B * SCl, NS * SCl
        )
        ctx.enqueue_copy(self.cd.dev.value(), sub_cd)
        ctx.enqueue_copy(self.cs.dev.value(), sub_cs)

        # ── imagination rollout (fully on device) ──
        for t in range(TI):
            ctx.enqueue_function[_feat_concat_k[NS, D, SCl]](
                self.cd.tt["gpu", NS * D](),
                self.cs.tt["gpu", NS * SCl](),
                self.fb.tt["gpu", NS * FEATl](),
                grid_dim=nbNF, block_dim=TPB,
            )
            ctx.enqueue_function[_hist_store_k[FEATl, TI, NS]](
                self.fb.tt["gpu", NS * FEATl](),
                self.feats_d.tt["gpu", NS * TI * FEATl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            policy.forward[target, NS](TensorRefs[1](self.fb), self.pb, ctx)
            ctx.enqueue_function[_cat_sample_hist_k[ACTD, TI, NS]](
                self.pb.tt["gpu", NS * ACTD](),
                self.noise_d.tt["gpu", TI * NS * ACTD](),
                self.at.tt["gpu", NS * ACTD](),
                self.pmean_d.tt["gpu", NS * TI * ACTD](),
                self.acts_d.tt["gpu", NS * TI * ACTD](),
                UNIMIX, Int64(t), grid_dim=nbB, block_dim=TPB,
            )
            value.forward[target, NS](TensorRefs[1](self.fb), self.vb, ctx)
            slowvalue.forward[target, NS](TensorRefs[1](self.fb), self.svb, ctx)
            ctx.enqueue_function[_hist_store_k[BINSl, TI, NS]](
                self.vb.tt["gpu", NS * BINSl](),
                self.vlog_d.tt["gpu", NS * TI * BINSl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            ctx.enqueue_function[_hist_store_k[BINSl, TI, NS]](
                self.svb.tt["gpu", NS * BINSl](),
                self.svlog_d.tt["gpu", NS * TI * BINSl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            rew.set_input["nd", NS](self.cd, ctx)
            rew.set_input["stoch_new", NS](self.cs, ctx)
            rew.set_input["rtgt", NS](self.dummy1, ctx)
            rew.forward[NS, target](self.dummy1, ctx)
            ref rew_logits = rew.node_output["rew"]()
            con.set_input["nd", NS](self.cd, ctx)
            con.set_input["stoch_new", NS](self.cs, ctx)
            con.set_input["ctgt", NS](self.dummy1, ctx)
            con.forward[NS, target](self.dummy1, ctx)
            ref con_logit = con.node_output["con"]()
            ctx.enqueue_function[_rewconv_hist_k[BINSl, TI, NS]](
                rew_logits.tt["gpu", NS * BINSl](),
                con_logit.tt["gpu", NS](),
                self.bins_d.tt["gpu", BINSl](),
                self.rewv_d.tt["gpu", NS * TI](),
                self.conv_d.tt["gpu", NS * TI](),
                Int64(t), grid_dim=nbBS, block_dim=BLKS,
            )
            imagine.set_input["deter", NS](self.cd, ctx)
            imagine.set_input["stoch", NS](self.cs, ctx)
            imagine.set_input["action", NS](self.at, ctx)
            imagine.forward[NS, target](self.fb, ctx)
            ref nd = imagine.node_output["nd"]()
            ref sn = imagine.node_output["stoch_new"]()
            ctx.enqueue_copy(self.cd.dev.value(), nd.dev.value())
            ctx.enqueue_copy(self.cs.dev.value(), sn.dev.value())

        # ── λ-return on device → ret_d ──
        ctx.enqueue_function[_imag_ret_k[NS, TI, BINSl]](
            self.vlog_d.tt["gpu", NS * TI * BINSl](),
            self.svlog_d.tt["gpu", NS * TI * BINSl](),
            self.rewv_d.tt["gpu", NS * TI](),
            self.conv_d.tt["gpu", NS * TI](),
            self.bins_d.tt["gpu", BINSl](),
            self.ret_d.tt["gpu", NS * TM1](),
            self.lam, Int32(1) if self.slowtar else Int32(0),
            grid_dim=nbBS, block_dim=BLKS,
        )
        # ── device-resident percentile retnorm (NO D2H — capture-safe) ──
        # Constant floor/frac indices (perclo/perchi over a fixed-size sample)
        # computed host-side; the percentile + EMA + rscale run on-device into
        # the persistent `retstate_d`, read by `_imag_bwd_k` via `rscale_d`.
        comptime NRET = NS * TM1
        comptime nbRET = (NRET + TPB - 1) // TPB
        var idx_lo = (retnorm.perclo / Scalar[DT](100.0)) * Scalar[DT](NRET - 1)
        var idx_hi = (retnorm.perchi / Scalar[DT](100.0)) * Scalar[DT](NRET - 1)
        var lo_floor = Int(idx_lo)
        var hi_floor = Int(idx_hi)
        ctx.enqueue_function[_ret_perc_neigh_k[NRET]](
            self.ret_d.tt["gpu", NRET](),
            self.neigh_d.tt["gpu", 4](),
            Int64(lo_floor), Int64(hi_floor), grid_dim=nbRET, block_dim=TPB,
        )
        ctx.enqueue_function[_ret_perc_ema_k](
            self.neigh_d.tt["gpu", 4](),
            self.retstate_d.tt["gpu", 2](),
            self.rscale_d.tt["gpu", 1](),
            idx_lo - Scalar[DT](lo_floor), idx_hi - Scalar[DT](hi_floor),
            retnorm.rate, retnorm.limit, grid_dim=1, block_dim=1,
        )

        # ── imag-loss backward (grads + per-(b,t) losses) on device ──
        var inv_im = Scalar[DT](1.0) / Scalar[DT](NS * TM1)
        ctx.enqueue_function[_imag_bwd_k[NS, TI, BINSl, ACTD]](
            self.vlog_d.tt["gpu", NS * TI * BINSl](),
            self.svlog_d.tt["gpu", NS * TI * BINSl](),
            self.pmean_d.tt["gpu", NS * TI * ACTD](),
            self.acts_d.tt["gpu", NS * TI * ACTD](),
            self.conv_d.tt["gpu", NS * TI](),
            self.ret_d.tt["gpu", NS * TM1](),
            self.bins_d.tt["gpu", BINSl](),
            self.gvlog_d.tt["gpu", NS * TI * BINSl](),
            self.gpmean_d.tt["gpu", NS * TI * ACTD](),
            self.polloss_d.tt["gpu", NS * TM1](),
            self.valloss_d.tt["gpu", NS * TM1](),
            self.rscale_d.tt["gpu", 1](),
            self.lam, self.actent, self.slowreg, inv_im, UNIMIX,
            grid_dim=nbBS, block_dim=BLKS,
        )

        # ── per-step value/policy vjp (inputs gathered from device histories) ──
        oval.zero_grad[target, M=Self.ValT](value, ctx)
        opol.zero_grad[target, M=Self.PolT](policy, ctx)
        for t in range(TI):
            ctx.enqueue_function[_hist_load_k[FEATl, TI, NS]](
                self.feats_d.tt["gpu", NS * TI * FEATl](),
                self.ftt.tt["gpu", NS * FEATl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            value.forward[target, NS](TensorRefs[1](self.ftt), self.vscr, ctx)
            ctx.enqueue_function[_hist_load_k[BINSl, TI, NS]](
                self.gvlog_d.tt["gpu", NS * TI * BINSl](),
                self.gvt.tt["gpu", NS * BINSl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            value.vjp[target, NS](
                TensorRefs[1](self.ftt), self.gvt, TensorRefs[1](self.gfeat), ctx
            )
            policy.forward[target, NS](TensorRefs[1](self.ftt), self.pscr, ctx)
            ctx.enqueue_function[_hist_load_k[ACTD, TI, NS]](
                self.gpmean_d.tt["gpu", NS * TI * ACTD](),
                self.polg.tt["gpu", NS * ACTD](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            policy.vjp[target, NS](
                TensorRefs[1](self.ftt), self.polg, TensorRefs[1](self.gfeat), ctx
            )

        # ── repval: ground value on REAL replay transitions (device) ──
        ctx.enqueue_function[_repval_setup_k[NS, TI, FEATl, Self.B, Self.T]](
            self.ret_d.tt["gpu", NS * TM1](),
            self.feats_d.tt["gpu", NS * TI * FEATl](),
            self.boot_d.tt["gpu", BT](),
            self.feat_bt.tt["gpu", BT * FEATl](),
            grid_dim=nbBT, block_dim=TPB,
        )
        value.forward[target, BT](TensorRefs[1](self.feat_bt), self.vlr, ctx)
        slowvalue.forward[target, BT](TensorRefs[1](self.feat_bt), self.svlr, ctx)
        var inv_rep = self.repval_scale / Scalar[DT](Self.B * TM1R)
        ctx.enqueue_function[_repl_bwd_k[Self.B, Self.T, BINSl]](
            self.mbdne_d.tt["gpu", BT](),
            self.mbrew_d.tt["gpu", BT](),
            self.boot_d.tt["gpu", BT](),
            self.svlr.tt["gpu", BT * BINSl](),
            self.bins_d.tt["gpu", BINSl](),
            self.g_vlr.tt["gpu", BT * BINSl](),
            self.vlr.tt["gpu", BT * BINSl](),
            self.horizon, self.lam, self.slowreg, inv_rep,
            grid_dim=nbB1, block_dim=TPB,
        )
        value.vjp[target, BT](
            TensorRefs[1](self.feat_bt), self.g_vlr, TensorRefs[1](self.grf), ctx
        )

        oval.step[target, M=Self.ValT](value, ctx)
        opol.step[target, M=Self.PolT](policy, ctx)
        polyak_module[target, Self.ValT](value, slowvalue, self.slow_rate, ctx=st.ctx)
        # End-of-step drain only matters when we read back below (the want_diag
        # block re-syncs after its D2H, and `last_ac_loss` is written there).
        # Gate it (Stage 3 P3d) → the discrete non-diag AC step issues no
        # sync/D2H at all: it is fully async, CUDA-graph capturable. Stream
        # ordering chains the next step's kernels after these enqueues.
        if want_diag:
            ctx.synchronize()

        # ── diagnostics + loss scalars (gated; log-cadence only) ──
        # Reduce every metric on-device into `diag_d` (single-block kernels),
        # then ONE tiny D2H of the whole bundle — no more full-history downloads
        # (feats_d 2.2 MB / vlog_d 0.8 MB / …). Slots:
        #   0 Σpolloss  1 Σvalloss  2 Σrewv  3 Σconv  4 min(conv)
        #   5 Σret  6 Σret²  7 Σ|pmean|  8 Σval  9 Σval²  10 Σfeats  11 Σfeats²
        if want_diag:
            comptime NTM = NS * TM1
            comptime NTI = NS * TI
            comptime NPM = NS * TI * ACTD
            comptime NFT = NS * TI * FEATl
            comptime r1 = 1  # single-block reductions
            ctx.enqueue_function[_diag_sum_k[NTM]](
                self.polloss_d.tt["gpu", NTM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(0), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_k[NTM]](
                self.valloss_d.tt["gpu", NTM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(1), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_k[NTI]](
                self.rewv_d.tt["gpu", NTI](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(2), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_k[NTI]](
                self.conv_d.tt["gpu", NTI](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(3), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_min_k[NTI]](
                self.conv_d.tt["gpu", NTI](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(4), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_sq_k[NTM]](
                self.ret_d.tt["gpu", NTM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(5), Int64(6), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_abs_sum_k[NPM]](
                self.pmean_d.tt["gpu", NPM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(7), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[
                _diag_twohot_sum_sq_k[NTI, BINSl, NS * TI * BINSl, BINSl]
            ](
                self.vlog_d.tt["gpu", NS * TI * BINSl](),
                self.bins_d.tt["gpu", BINSl](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(8), Int64(9), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_sq_k[NFT]](
                self.feats_d.tt["gpu", NFT](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(10), Int64(11), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            # per-action mean advantage (collapse-driver probe; serial, diag-only)
            ctx.enqueue_function[_diag_adv_act_k[NS, TI, BINSl, ACTD]](
                self.acts_d.tt["gpu", NS * TI * ACTD](),
                self.ret_d.tt["gpu", NS * TM1](),
                self.vlog_d.tt["gpu", NS * TI * BINSl](),
                self.bins_d.tt["gpu", BINSl](),
                self.rscale_d.tt["gpu", 1](),
                self.adv_diag_d.tt["gpu", 2 * ACTD](),
                grid_dim=1, block_dim=1,
            )
            self.diag_d.download(ctx)
            self.adv_diag_d.download(ctx)
            self.rscale_d.download(ctx)
            ctx.synchronize()
            st.dbg_rscale = self.rscale_d.data[0]
            var inv_ac = Scalar[DT](1.0) / Scalar[DT](NTM)
            var d0 = self.diag_d.data[0]
            var d1 = self.diag_d.data[1]
            st.last_ac_loss = (d0 + d1) * inv_ac
            st.dbg_pol_loss = d0 * inv_ac
            st.dbg_val_loss = d1 * inv_ac
            st.dbg_rew_pred = self.diag_d.data[2] / Scalar[DT](NTI)
            st.dbg_con_mean = self.diag_d.data[3] / Scalar[DT](NTI)
            st.dbg_con_min = self.diag_d.data[4]
            var rmean = self.diag_d.data[5] / Scalar[DT](NTM)
            st.dbg_ret_mean = rmean
            var rvar = self.diag_d.data[6] / Scalar[DT](NTM) - rmean * rmean
            st.dbg_ret_std = sqrt(rvar if rvar > Scalar[DT](0.0) else Scalar[DT](0.0))
            st.dbg_pmean_abs = self.diag_d.data[7] / Scalar[DT](NPM)
            st.dbg_pstd = 0.0
            var vmean = self.diag_d.data[8] / Scalar[DT](NTI)
            st.dbg_val_mean = vmean
            var vvar = self.diag_d.data[9] / Scalar[DT](NTI) - vmean * vmean
            st.dbg_val_std = sqrt(vvar if vvar > Scalar[DT](0.0) else Scalar[DT](0.0))
            var fmean = self.diag_d.data[10] / Scalar[DT](NFT)
            var fvar = self.diag_d.data[11] / Scalar[DT](NFT) - fmean * fmean
            st.dbg_feat_std = sqrt(fvar if fvar > Scalar[DT](0.0) else Scalar[DT](0.0))
            for a in range(ACTD):
                var acnt = self.adv_diag_d.data[ACTD + a]
                st.dbg_adv_act[a] = (
                    self.adv_diag_d.data[a] / acnt
                    if acnt > Scalar[DT](0.0) else Scalar[DT](0.0)
                )

    def _ac_gpu_cont[target: StaticString](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, Self.T_IMAG],
        mut imagine: Self.ImagT,
        mut value: Self.ValT,
        mut slowvalue: Self.ValT,
        mut policy: Self.PolT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oval: DreamerOpt,
        mut opol: DreamerOpt,
        mut retnorm: PercentileNormalize,
        bins: Pointer[Scalar[DT], MutAnyOrigin],
        want_diag: Bool,
    ) raises:
        # Device-resident CONTINUOUS imagination-AC (the bounded-normal twin of
        # `_ac_gpu_disc`): rollout + λ-return + imag/repl loss all run on-device
        # through the kernels, eliminating the OLD `_ac_gpu`'s ~7 host round-trips
        # per imagination step. Inputs are uploaded ONCE at the top (carries,
        # uniform noise, replay rew/dne, twohot bins) — host-sourced like the old
        # path (so NO trainer-gate changes), then the per-step compute is fully on
        # device. Not CUDA-graph captured (the top uploads are eager); `want_diag`
        # gates the host metric readout. Noise is uniform[-1,1] (matches `_ac_cpu`
        # → CPU↔GPU parity); λ-return is slowtar=False on device (as discrete).
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime TI = Self.T_IMAG
        comptime NS = Self.T * Self.B
        comptime TM1 = TI - 1
        comptime BT = Self.B * Self.T
        comptime TM1R = Self.T - 1
        comptime nbB = (NS + TPB - 1) // TPB
        comptime nbNF = (NS * FEATl + TPB - 1) // TPB  # one thread / hist element
        comptime BLKS = 32  # one warp = 32 starts
        comptime nbBS = (NS + BLKS - 1) // BLKS  # spread NS starts over SMs
        comptime nbBT = (BT + TPB - 1) // TPB
        comptime nbB1 = (Self.B + TPB - 1) // TPB
        var ctx = st.ctx.value()
        var MINSTD = self.minstd
        var MAXSTD = self.maxstd

        # ── device-direct inputs (capture-safe; NO H2D in this region) ──
        # Mirrors `_ac_gpu_disc`: bins are uploaded once in DreamerV3Trainer.make,
        # the uniform imagination noise by the trainer `_fill_noise` prologue, and
        # rew/dne + carries come device-direct from the shared DreamerState buffers
        # that `_draw_minibatch`/`load_minibatch` (rew/dne → d_rew/d_cont) and
        # `_wm_gpu` (carry → d_cdeter/d_cstoch) left there. So this whole method is
        # free of uploads/downloads/syncs (except the want_diag-gated diag) and can
        # be replayed from a captured CUDA graph (Stage 3; parity-gated).
        ctx.enqueue_copy(self.mbrew_d.dev.value(), st.d_rew.dev.value())
        ctx.enqueue_copy(self.mbdne_d.dev.value(), st.d_cont.dev.value())
        # init rollout carry cd/cs from posterior carries 1..T — device-direct
        # from the WM scan (skip the index-0 init carry: offset B*D / B*SC,
        # length NS*D / NS*SC) straight into cd/cs.
        var sub_cd = st.d_cdeter.dev.value().create_sub_buffer[DT](
            Self.B * D, NS * D
        )
        var sub_cs = st.d_cstoch.dev.value().create_sub_buffer[DT](
            Self.B * SCl, NS * SCl
        )
        ctx.enqueue_copy(self.cd.dev.value(), sub_cd)
        ctx.enqueue_copy(self.cs.dev.value(), sub_cs)

        # ── imagination rollout (fully on device) ──
        for t in range(TI):
            ctx.enqueue_function[_feat_concat_k[NS, D, SCl]](
                self.cd.tt["gpu", NS * D](),
                self.cs.tt["gpu", NS * SCl](),
                self.fb.tt["gpu", NS * FEATl](),
                grid_dim=nbNF, block_dim=TPB,
            )
            ctx.enqueue_function[_hist_store_k[FEATl, TI, NS]](
                self.fb.tt["gpu", NS * FEATl](),
                self.feats_d.tt["gpu", NS * TI * FEATl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            policy.forward[target, NS](TensorRefs[1](self.fb), self.pb, ctx)
            ctx.enqueue_function[_gaussian_sample_hist_k[ACTD, TI, NS]](
                self.pb.tt["gpu", NS * 2 * ACTD](),
                self.noise_d.tt["gpu", TI * NS * ACTD](),
                self.at.tt["gpu", NS * ACTD](),
                self.pmean_d.tt["gpu", NS * TI * ACTD](),
                self.pstd_d.tt["gpu", NS * TI * ACTD](),
                self.acts_d.tt["gpu", NS * TI * ACTD](),
                MINSTD, MAXSTD, Int64(t), grid_dim=nbB, block_dim=TPB,
            )
            value.forward[target, NS](TensorRefs[1](self.fb), self.vb, ctx)
            slowvalue.forward[target, NS](TensorRefs[1](self.fb), self.svb, ctx)
            ctx.enqueue_function[_hist_store_k[BINSl, TI, NS]](
                self.vb.tt["gpu", NS * BINSl](),
                self.vlog_d.tt["gpu", NS * TI * BINSl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            ctx.enqueue_function[_hist_store_k[BINSl, TI, NS]](
                self.svb.tt["gpu", NS * BINSl](),
                self.svlog_d.tt["gpu", NS * TI * BINSl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            rew.set_input["nd", NS](self.cd, ctx)
            rew.set_input["stoch_new", NS](self.cs, ctx)
            rew.set_input["rtgt", NS](self.dummy1, ctx)
            rew.forward[NS, target](self.dummy1, ctx)
            ref rew_logits = rew.node_output["rew"]()
            con.set_input["nd", NS](self.cd, ctx)
            con.set_input["stoch_new", NS](self.cs, ctx)
            con.set_input["ctgt", NS](self.dummy1, ctx)
            con.forward[NS, target](self.dummy1, ctx)
            ref con_logit = con.node_output["con"]()
            ctx.enqueue_function[_rewconv_hist_k[BINSl, TI, NS]](
                rew_logits.tt["gpu", NS * BINSl](),
                con_logit.tt["gpu", NS](),
                self.bins_d.tt["gpu", BINSl](),
                self.rewv_d.tt["gpu", NS * TI](),
                self.conv_d.tt["gpu", NS * TI](),
                Int64(t), grid_dim=nbBS, block_dim=BLKS,
            )
            imagine.set_input["deter", NS](self.cd, ctx)
            imagine.set_input["stoch", NS](self.cs, ctx)
            imagine.set_input["action", NS](self.at, ctx)
            imagine.forward[NS, target](self.fb, ctx)
            ref nd = imagine.node_output["nd"]()
            ref sn = imagine.node_output["stoch_new"]()
            ctx.enqueue_copy(self.cd.dev.value(), nd.dev.value())
            ctx.enqueue_copy(self.cs.dev.value(), sn.dev.value())

        # ── λ-return on device → ret_d ──
        ctx.enqueue_function[_imag_ret_k[NS, TI, BINSl]](
            self.vlog_d.tt["gpu", NS * TI * BINSl](),
            self.svlog_d.tt["gpu", NS * TI * BINSl](),
            self.rewv_d.tt["gpu", NS * TI](),
            self.conv_d.tt["gpu", NS * TI](),
            self.bins_d.tt["gpu", BINSl](),
            self.ret_d.tt["gpu", NS * TM1](),
            self.lam, Int32(1) if self.slowtar else Int32(0),
            grid_dim=nbBS, block_dim=BLKS,
        )
        # ── device-resident percentile retnorm (no D2H) ──
        comptime NRET = NS * TM1
        comptime nbRET = (NRET + TPB - 1) // TPB
        var idx_lo = (retnorm.perclo / Scalar[DT](100.0)) * Scalar[DT](NRET - 1)
        var idx_hi = (retnorm.perchi / Scalar[DT](100.0)) * Scalar[DT](NRET - 1)
        var lo_floor = Int(idx_lo)
        var hi_floor = Int(idx_hi)
        ctx.enqueue_function[_ret_perc_neigh_k[NRET]](
            self.ret_d.tt["gpu", NRET](),
            self.neigh_d.tt["gpu", 4](),
            Int64(lo_floor), Int64(hi_floor), grid_dim=nbRET, block_dim=TPB,
        )
        ctx.enqueue_function[_ret_perc_ema_k](
            self.neigh_d.tt["gpu", 4](),
            self.retstate_d.tt["gpu", 2](),
            self.rscale_d.tt["gpu", 1](),
            idx_lo - Scalar[DT](lo_floor), idx_hi - Scalar[DT](hi_floor),
            retnorm.rate, retnorm.limit, grid_dim=1, block_dim=1,
        )

        # ── imag-loss backward (Gaussian) on device ──
        var inv_im = Scalar[DT](1.0) / Scalar[DT](NS * TM1)
        # Pre-zero the grad buffers: gvlog ACCUMULATES (two twohot-CE passes) and
        # the t=TM1 slice of all three is never written by the backward → must be
        # 0. Grid-strided, then the per-(start,t) kernel fills t<TM1.
        comptime NTB = NS * TI * BINSl
        comptime NTA = NS * TI * ACTD
        ctx.enqueue_function[_zero_k[NTB]](
            self.gvlog_d.tt["gpu", NTB](),
            grid_dim=(NTB + TPB - 1) // TPB, block_dim=TPB,
        )
        ctx.enqueue_function[_zero_k[NTA]](
            self.gpmean_d.tt["gpu", NTA](),
            grid_dim=(NTA + TPB - 1) // TPB, block_dim=TPB,
        )
        ctx.enqueue_function[_zero_k[NTA]](
            self.gpstd_d.tt["gpu", NTA](),
            grid_dim=(NTA + TPB - 1) // TPB, block_dim=TPB,
        )
        # one thread per (start, imagination step) over NS*(TI-1) — full occupancy
        comptime nbPT = (NS * TM1 + TPB - 1) // TPB
        ctx.enqueue_function[_imag_bwd_continuous_pt_k[NS, TI, BINSl, ACTD]](
            self.vlog_d.tt["gpu", NS * TI * BINSl](),
            self.svlog_d.tt["gpu", NS * TI * BINSl](),
            self.pmean_d.tt["gpu", NS * TI * ACTD](),
            self.pstd_d.tt["gpu", NS * TI * ACTD](),
            self.acts_d.tt["gpu", NS * TI * ACTD](),
            self.conv_d.tt["gpu", NS * TI](),
            self.ret_d.tt["gpu", NS * TM1](),
            self.bins_d.tt["gpu", BINSl](),
            self.gvlog_d.tt["gpu", NS * TI * BINSl](),
            self.gpmean_d.tt["gpu", NS * TI * ACTD](),
            self.gpstd_d.tt["gpu", NS * TI * ACTD](),
            self.polloss_d.tt["gpu", NS * TM1](),
            self.valloss_d.tt["gpu", NS * TM1](),
            self.rscale_d.tt["gpu", 1](),
            self.actent, self.slowreg, inv_im, MINSTD, MAXSTD,
            grid_dim=nbPT, block_dim=TPB,
        )

        # ── per-step value/policy vjp (inputs gathered from device histories) ──
        oval.zero_grad[target, M=Self.ValT](value, ctx)
        opol.zero_grad[target, M=Self.PolT](policy, ctx)
        for t in range(TI):
            ctx.enqueue_function[_hist_load_k[FEATl, TI, NS]](
                self.feats_d.tt["gpu", NS * TI * FEATl](),
                self.ftt.tt["gpu", NS * FEATl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            value.forward[target, NS](TensorRefs[1](self.ftt), self.vscr, ctx)
            ctx.enqueue_function[_hist_load_k[BINSl, TI, NS]](
                self.gvlog_d.tt["gpu", NS * TI * BINSl](),
                self.gvt.tt["gpu", NS * BINSl](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            value.vjp[target, NS](
                TensorRefs[1](self.ftt), self.gvt, TensorRefs[1](self.gfeat), ctx
            )
            policy.forward[target, NS](TensorRefs[1](self.ftt), self.pscr, ctx)
            ctx.enqueue_function[_hist_load_pol2_k[ACTD, TI, NS]](
                self.gpmean_d.tt["gpu", NS * TI * ACTD](),
                self.gpstd_d.tt["gpu", NS * TI * ACTD](),
                self.polg.tt["gpu", NS * 2 * ACTD](),
                Int64(t), grid_dim=nbNF, block_dim=TPB,
            )
            policy.vjp[target, NS](
                TensorRefs[1](self.ftt), self.polg, TensorRefs[1](self.gfeat), ctx
            )

        # ── repval: ground value on REAL replay transitions (device) ──
        ctx.enqueue_function[_repval_setup_k[NS, TI, FEATl, Self.B, Self.T]](
            self.ret_d.tt["gpu", NS * TM1](),
            self.feats_d.tt["gpu", NS * TI * FEATl](),
            self.boot_d.tt["gpu", BT](),
            self.feat_bt.tt["gpu", BT * FEATl](),
            grid_dim=nbBT, block_dim=TPB,
        )
        value.forward[target, BT](TensorRefs[1](self.feat_bt), self.vlr, ctx)
        slowvalue.forward[target, BT](TensorRefs[1](self.feat_bt), self.svlr, ctx)
        var inv_rep = self.repval_scale / Scalar[DT](Self.B * TM1R)
        ctx.enqueue_function[_repl_bwd_k[Self.B, Self.T, BINSl]](
            self.mbdne_d.tt["gpu", BT](),
            self.mbrew_d.tt["gpu", BT](),
            self.boot_d.tt["gpu", BT](),
            self.svlr.tt["gpu", BT * BINSl](),
            self.bins_d.tt["gpu", BINSl](),
            self.g_vlr.tt["gpu", BT * BINSl](),
            self.vlr.tt["gpu", BT * BINSl](),
            self.horizon, self.lam, self.slowreg, inv_rep,
            grid_dim=nbB1, block_dim=TPB,
        )
        value.vjp[target, BT](
            TensorRefs[1](self.feat_bt), self.g_vlr, TensorRefs[1](self.grf), ctx
        )

        oval.step[target, M=Self.ValT](value, ctx)
        opol.step[target, M=Self.PolT](policy, ctx)
        polyak_module[target, Self.ValT](value, slowvalue, self.slow_rate, ctx=st.ctx)

        # ── diagnostics + loss scalars (gated; log-cadence only) ──
        if want_diag:
            ctx.synchronize()
            comptime NTM = NS * TM1
            comptime NTI = NS * TI
            comptime NPM = NS * TI * ACTD
            comptime NFT = NS * TI * FEATl
            comptime r1 = 1
            ctx.enqueue_function[_diag_sum_k[NTM]](
                self.polloss_d.tt["gpu", NTM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(0), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_k[NTM]](
                self.valloss_d.tt["gpu", NTM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(1), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_k[NTI]](
                self.rewv_d.tt["gpu", NTI](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(2), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_k[NTI]](
                self.conv_d.tt["gpu", NTI](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(3), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_min_k[NTI]](
                self.conv_d.tt["gpu", NTI](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(4), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_sq_k[NTM]](
                self.ret_d.tt["gpu", NTM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(5), Int64(6), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_abs_sum_k[NPM]](
                self.pmean_d.tt["gpu", NPM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(7), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[
                _diag_twohot_sum_sq_k[NTI, BINSl, NS * TI * BINSl, BINSl]
            ](
                self.vlog_d.tt["gpu", NS * TI * BINSl](),
                self.bins_d.tt["gpu", BINSl](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(8), Int64(9), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            ctx.enqueue_function[_diag_sum_sq_k[NFT]](
                self.feats_d.tt["gpu", NFT](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(10), Int64(11), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            # slot 12: Σ std_raw → dbg_pstd (raw; continuous-only diagnostic)
            ctx.enqueue_function[_diag_sum_k[NPM]](
                self.pstd_d.tt["gpu", NPM](),
                self.diag_d.tt["gpu", DIAG_N](),
                Int64(12), grid_dim=r1, block_dim=TPB_REDUCE,
            )
            self.diag_d.download(ctx)
            self.rscale_d.download(ctx)
            ctx.synchronize()
            st.dbg_rscale = self.rscale_d.data[0]
            var inv_ac = Scalar[DT](1.0) / Scalar[DT](NTM)
            var d0 = self.diag_d.data[0]
            var d1 = self.diag_d.data[1]
            st.last_ac_loss = (d0 + d1) * inv_ac
            st.dbg_pol_loss = d0 * inv_ac
            st.dbg_val_loss = d1 * inv_ac
            st.dbg_rew_pred = self.diag_d.data[2] / Scalar[DT](NTI)
            st.dbg_con_mean = self.diag_d.data[3] / Scalar[DT](NTI)
            st.dbg_con_min = self.diag_d.data[4]
            var rmean = self.diag_d.data[5] / Scalar[DT](NTM)
            st.dbg_ret_mean = rmean
            var rvar = self.diag_d.data[6] / Scalar[DT](NTM) - rmean * rmean
            st.dbg_ret_std = sqrt(rvar if rvar > Scalar[DT](0.0) else Scalar[DT](0.0))
            st.dbg_pmean_abs = self.diag_d.data[7] / Scalar[DT](NPM)
            st.dbg_pstd = self.diag_d.data[12] / Scalar[DT](NPM)
            var vmean = self.diag_d.data[8] / Scalar[DT](NTI)
            st.dbg_val_mean = vmean
            var vvar = self.diag_d.data[9] / Scalar[DT](NTI) - vmean * vmean
            st.dbg_val_std = sqrt(vvar if vvar > Scalar[DT](0.0) else Scalar[DT](0.0))
            var fmean = self.diag_d.data[10] / Scalar[DT](NFT)
            var fvar = self.diag_d.data[11] / Scalar[DT](NFT) - fmean * fmean
            st.dbg_feat_std = sqrt(fvar if fvar > Scalar[DT](0.0) else Scalar[DT](0.0))
