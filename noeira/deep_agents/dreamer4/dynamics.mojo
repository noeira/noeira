"""Dreamer4Dynamics — interactive dynamics model (model.py:Dynamics).

The dynamics transformer runs the *same* block-causal backbone as the
tokenizer, but over an interleaved per-frame token sequence

    [ action | signal | step | spatial(n_spatial) | register(n_register) ]

(the agent tokens of §3.3 are added in Phase 3; here `n_agent = 0`). With no
latents and no agent tokens, the `wm_agent_isolated` space mask reduces to
**full mixing** (`"wm_agent"`) and `latents_only_time=False` makes the time
layer attend over *all* S tokens — which `TimeAttentionLatents[D,NH,T,S, L=S]`
already gives (every token treated as a latent ⇒ no zeroing). So the backbone
is `Dreamer4Stack[..., L=S, MODE="wm_agent"]` with **no new block code**.

The module splits into a bespoke **front-end** (assemble the S-token grid from
the spatial projection, the learned action base, the step/signal embedding
tables, and the learned register tokens) and a pure-combinator **tail**
(`SinusoidalPosAddBT → Dreamer4Stack → Slice(spatial) → Tokenwise[ZeroLinear]`).
The tail owns the whole transformer + the zero-init `flow_x_head` (ZeroLinear
⇒ x-prediction starts at 0, the shortcut-forcing fixed point). The front-end
hand-rolls the concat/gather and its vjp.

Driven directly by the shortcut-forcing loss (not a Trainer). The per-sample
signal/step indices are *host-known control inputs*, so they are pushed via
`set_indices(sig, step, BATCH)` before each forward rather than threaded as
heterogeneous tensor inputs:
    set_indices(sig, step)            sig/step : [B·T] fp buffers (exact ints)
    forward(input=z̃)                  z̃        : [B·T, n_spatial·d_spatial]
    output = x̂1 (x-prediction)         [B·T, n_spatial·d_spatial]

ACTION CONDITIONING (model.py:ActionEncoder): gated by the `ADIM` comptime
param. ADIM=0 (default) = **unconditional** — the action token is the learned
`action_base` only (reference's `actions is None` path); everything is
byte-for-byte the original dynamics. ADIM>0 adds an action token
`action_base + act_mlp(clamp(act_mask ⊙ a, -1, 1))`, where
`act_mlp = Linear[ADIM,AHID] → SiLU → ZeroLinear[AHID,D]`. The ZeroLinear
second layer makes the action contribution start EXACTLY 0, so at init a
conditioned model equals the unconditional one (the reference approximates
this with fc2 std=1e-3; ZeroLinear is exact). Actions are pushed via
`set_actions(actions, act_mask, batch)` (a host/device control input like
`set_indices`, since they're data — the clamp/mask carry no gradient and the
act-MLP's grad_input is discarded). CPU + GPU; the act token grad (= the
action_base grad) is the act-MLP's grad_output. Only the action token changes
— the loss and backbone are untouched.

PHASE 2: CPU forward + vjp (2.2). GPU forward + vjp (2.4) — the tail
(Sequential) is already GPU-capable; the bespoke front-end gets device
kernels for token assembly + its vjp (grad-input gather + four param-grad
kernels). The signal/step embedding-table grads use index-masked batch
reductions (one thread per (vocab_row, channel), looping the small batch) so
no atomics are needed — the signal/step vocabs are tiny (KMAX+1, log2 KMAX+1).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import Param, ParamVisitor
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.walkers import (
    for_each_param_auto, zero_grad_auto, join_name,
)
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.tokenwise import Tokenwise
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.silu import SiLU
from noeira.nn.primitives.slice import Slice
from noeira.nn.primitives.zero_linear import ZeroLinear
from noeira.nn.primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from .blocks import Dreamer4Stack, Dreamer4DynStack
from .shortcut_loss import ShortcutDynamics, AgentDynamics, _mao


def _ilog2(n: Int) -> Int:
    var k = 0
    var v = n
    while v > 1:
        v //= 2
        k += 1
    return k


def _pos(a: Int, fallback: Int) -> Int:
    """Comptime helper: `a` if positive else `fallback` (avoids the
    conditional-type-alias footgun by keeping the ternary at Int level)."""
    return a if a > 0 else fallback


# ── GPU front-end kernels ───────────────────────────────────────────────
# Assemble the (B·T, S·D) token grid: [action | signal | step | spatial | reg].
def _dyn_assemble_kernel[
    BT: Int, S: Int, D: Int, NSP: Int, NREG: Int, NSIG: Int, NSTEP: Int
](
    proj_out: Tile[DT, BT * NSP * D],
    action_base: Tile[DT, D],
    signal_table: Tile[DT, NSIG * D],
    step_table: Tile[DT, NSTEP * D],
    register: Tile[DT, NREG * D],
    sig: Tile[DT, BT],
    step: Tile[DT, BT],
    grid: Tile[DT, BT * S * D],
):
    var idx = Int(global_idx.x)
    if idx >= BT * S * D:
        return
    comptime SD = S * D
    var bt = idx // SD
    var col = idx % SD
    var token = col // D
    var d = col % D
    if token == 0:
        grid.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](action_base.ptr[unsafe_offset=d])
    elif token == 1:
        var si = Int(rebind[Scalar[DT]](sig.ptr[unsafe_offset=bt]) + Scalar[DT](0.5))
        grid.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](signal_table.ptr[unsafe_offset=si * D + d])
    elif token == 2:
        var pi = Int(rebind[Scalar[DT]](step.ptr[unsafe_offset=bt]) + Scalar[DT](0.5))
        grid.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](step_table.ptr[unsafe_offset=pi * D + d])
    elif token < 3 + NSP:
        grid.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
            proj_out.ptr[unsafe_offset=bt * (NSP * D) + (token - 3) * D + d]
        )
    elif token < 3 + NSP + NREG:
        grid.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](
            register.ptr[unsafe_offset=(token - (3 + NSP)) * D + d]
        )
    # else: agent token columns (token ≥ 3+NSP+NREG) are written separately by
    # `_dyn_set_agent_kernel` when AGENT; with NAGENT=0 there are no such cols.


# grad_grid → grad_proj_out (spatial tokens at cols [3D, (3+NSP)D)).
def _dyn_grad_proj_kernel[BT: Int, S: Int, D: Int, NSP: Int](
    ggrid: Tile[DT, BT * S * D],
    gproj: Tile[DT, BT * NSP * D],
):
    var idx = Int(global_idx.x)
    if idx >= BT * NSP * D:
        return
    var bt = idx // (NSP * D)
    var k = idx % (NSP * D)
    gproj.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](ggrid.ptr[unsafe_offset=bt * (S * D) + 3 * D + k])


# action_base grad: gbase[d] += Σ_bt ggrid[bt, d].
def _dyn_grad_base_kernel[BT: Int, S: Int, D: Int](
    ggrid: Tile[DT, BT * S * D],
    gbase: Tile[DT, D],
):
    var d = Int(global_idx.x)
    if d >= D:
        return
    var acc = Scalar[DT](0.0)
    for bt in range(BT):
        acc += rebind[Scalar[DT]](ggrid.ptr[unsafe_offset=bt * (S * D) + d])
    gbase.ptr[unsafe_offset=d] = rebind[Scalar[DT]](gbase.ptr[unsafe_offset=d]) + acc


# register grad: greg[k] += Σ_bt ggrid[bt, REG_OFF + k].
def _dyn_grad_reg_kernel[BT: Int, S: Int, D: Int, NSP: Int, NREG: Int](
    ggrid: Tile[DT, BT * S * D],
    greg: Tile[DT, NREG * D],
):
    var k = Int(global_idx.x)
    if k >= NREG * D:
        return
    comptime REG_OFF = (3 + NSP) * D
    var acc = Scalar[DT](0.0)
    for bt in range(BT):
        acc += rebind[Scalar[DT]](ggrid.ptr[unsafe_offset=bt * (S * D) + REG_OFF + k])
    greg.ptr[unsafe_offset=k] = rebind[Scalar[DT]](greg.ptr[unsafe_offset=k]) + acc


# signal-table grad (index-masked batch reduction; signal token at col [D,2D)).
def _dyn_grad_sig_kernel[BT: Int, S: Int, D: Int, NSIG: Int](
    ggrid: Tile[DT, BT * S * D],
    sig: Tile[DT, BT],
    gsig: Tile[DT, NSIG * D],
):
    var e = Int(global_idx.x)
    if e >= NSIG * D:
        return
    var v = e // D
    var d = e % D
    var acc = Scalar[DT](0.0)
    for bt in range(BT):
        var si = Int(rebind[Scalar[DT]](sig.ptr[unsafe_offset=bt]) + Scalar[DT](0.5))
        if si == v:
            acc += rebind[Scalar[DT]](ggrid.ptr[unsafe_offset=bt * (S * D) + D + d])
    gsig.ptr[unsafe_offset=e] = rebind[Scalar[DT]](gsig.ptr[unsafe_offset=e]) + acc


# step-table grad (index-masked; step token at col [2D,3D)).
def _dyn_grad_step_kernel[BT: Int, S: Int, D: Int, NSTEP: Int](
    ggrid: Tile[DT, BT * S * D],
    step: Tile[DT, BT],
    gstep: Tile[DT, NSTEP * D],
):
    var e = Int(global_idx.x)
    if e >= NSTEP * D:
        return
    var v = e // D
    var d = e % D
    var acc = Scalar[DT](0.0)
    for bt in range(BT):
        var pi = Int(rebind[Scalar[DT]](step.ptr[unsafe_offset=bt]) + Scalar[DT](0.5))
        if pi == v:
            acc += rebind[Scalar[DT]](ggrid.ptr[unsafe_offset=bt * (S * D) + 2 * D + d])
    gstep.ptr[unsafe_offset=e] = rebind[Scalar[DT]](gstep.ptr[unsafe_offset=e]) + acc


# action conditioning: add the act-MLP output into the action token (col [0,D))
# of the assembled grid — fires only when ACOND (one thread per (bt,d)).
def _dyn_add_act_kernel[BT: Int, S: Int, D: Int](
    aout: Tile[DT, BT * D],
    grid: Tile[DT, BT * S * D],
):
    var e = Int(global_idx.x)
    if e >= BT * D:
        return
    var bt = e // D
    var d = e % D
    grid.ptr[unsafe_offset=bt * (S * D) + d] = (
        rebind[Scalar[DT]](grid.ptr[unsafe_offset=bt * (S * D) + d])
        + rebind[Scalar[DT]](aout.ptr[unsafe_offset=e])
    )


# action conditioning vjp: extract the action-token grad (col [0,D)) into a
# packed [BT, D] buffer = the act-MLP's grad_output (same grad as action_base).
def _dyn_extract_token0_kernel[BT: Int, S: Int, D: Int](
    ggrid: Tile[DT, BT * S * D],
    gaout: Tile[DT, BT * D],
):
    var e = Int(global_idx.x)
    if e >= BT * D:
        return
    var bt = e // D
    var d = e % D
    gaout.ptr[unsafe_offset=e] = rebind[Scalar[DT]](ggrid.ptr[unsafe_offset=bt * (S * D) + d])


# ── Agent-token GPU kernels (AGENT only). The agent columns start at
# (S - NAGENT)·D = AGENT_OFF; AG_DIM = NAGENT·D. One thread per (bt, k). ──
# write the agent input into the grid agent columns.
def _dyn_set_agent_kernel[BT: Int, S: Int, D: Int, NAGENT: Int](
    agin: Tile[DT, BT * NAGENT * D],
    grid: Tile[DT, BT * S * D],
):
    comptime AG = NAGENT * D
    comptime OFF = (S - NAGENT) * D
    var e = Int(global_idx.x)
    if e >= BT * AG:
        return
    var bt = e // AG
    var k = e % AG
    grid.ptr[unsafe_offset=bt * (S * D) + OFF + k] = rebind[Scalar[DT]](agin.ptr[unsafe_offset=e])


# extract h_t = transformer-output agent columns into a packed [BT, AG] buffer.
def _dyn_extract_agent_fwd_kernel[BT: Int, S: Int, D: Int, NAGENT: Int](
    tfout: Tile[DT, BT * S * D],
    agout: Tile[DT, BT * NAGENT * D],
):
    comptime AG = NAGENT * D
    comptime OFF = (S - NAGENT) * D
    var e = Int(global_idx.x)
    if e >= BT * AG:
        return
    var bt = e // AG
    var k = e % AG
    agout.ptr[unsafe_offset=e] = rebind[Scalar[DT]](tfout.ptr[unsafe_offset=bt * (S * D) + OFF + k])


# add the h_t grad into the agent columns of the transformer-output grad.
def _dyn_add_agent_grad_kernel[BT: Int, S: Int, D: Int, NAGENT: Int](
    gagout: Tile[DT, BT * NAGENT * D],
    gtfout: Tile[DT, BT * S * D],
):
    comptime AG = NAGENT * D
    comptime OFF = (S - NAGENT) * D
    var e = Int(global_idx.x)
    if e >= BT * AG:
        return
    var bt = e // AG
    var k = e % AG
    gtfout.ptr[unsafe_offset=bt * (S * D) + OFF + k] = (
        rebind[Scalar[DT]](gtfout.ptr[unsafe_offset=bt * (S * D) + OFF + k])
        + rebind[Scalar[DT]](gagout.ptr[unsafe_offset=e])
    )


# extract the grad wrt the agent input from the grid grad agent columns.
def _dyn_extract_agent_grad_kernel[BT: Int, S: Int, D: Int, NAGENT: Int](
    ggrid: Tile[DT, BT * S * D],
    gagin: Tile[DT, BT * NAGENT * D],
):
    comptime AG = NAGENT * D
    comptime OFF = (S - NAGENT) * D
    var e = Int(global_idx.x)
    if e >= BT * AG:
        return
    var bt = e // AG
    var k = e % AG
    gagin.ptr[unsafe_offset=e] = rebind[Scalar[DT]](ggrid.ptr[unsafe_offset=bt * (S * D) + OFF + k])


struct Dreamer4Dynamics[
    DSP: Int,      # d_spatial (packed bottleneck width per spatial token)
    NSP: Int,      # n_spatial
    D: Int,        # d_model
    NH: Int,       # n_heads
    T: Int,        # frames per window
    NREG: Int,     # n_register
    HID: Int,      # SwiGLU hidden
    DEPTH: Int,    # transformer depth
    KMAX: Int,     # k_max (max integration steps; power of two)
    USE_MAX: Bool = True,
    ADIM: Int = 0,   # action dim (0 ⇒ unconditional: learned base token only)
    AHID: Int = 0,   # action-MLP hidden (0 ⇒ derive 2·D, matching the reference)
    NAGENT: Int = 0, # agent tokens (0 ⇒ no agent modality; Phase 3 BC uses >0)
](Module, AgentDynamics):
    comptime ARITY: Int = 1
    # tokens per frame: [action|signal|step|spatial×NSP|register×NREG|agent×NAGENT]
    comptime S: Int = 3 + Self.NSP + Self.NREG + Self.NAGENT
    comptime NSIG: Int = Self.KMAX + 1                  # signal vocab
    comptime NSTEP: Int = _ilog2(Self.KMAX) + 1         # step vocab (num bins)
    comptime SD: Int = Self.S * Self.D
    comptime SPAT_OFF: Int = 3 * Self.D                 # spatial token col start
    comptime REG_OFF: Int = (3 + Self.NSP) * Self.D     # register col start
    comptime AGENT_OFF: Int = (3 + Self.NSP + Self.NREG) * Self.D  # agent col start
    comptime IN_DIMS = Array[Int, 1](fill=Self.NSP * Self.DSP)
    comptime OUT_DIM = Self.NSP * Self.DSP
    # `Module` and `ShortcutDynamics` both default ACT_DT=DT → resolve manually.
    comptime ACT_DT = DT

    # Agent conditioning (model.py §3.3 + paper eq. 9). AGENT gates the whole
    # agent-token path; when off (NAGENT=0) the token layout has no agent
    # modality, the `wm_agent_bc` space mask collapses to full mixing, and
    # everything below is byte-for-byte the unconditional dynamics. AG_DIM is
    # the total agent-token width (NAGENT·D); AG_DIM_EFF keeps size-dependent
    # buffers valid (≥1) when unused. The agent token INPUT is pushed via
    # `set_agent_in` (the TaskEmbedder output, a real differentiable input);
    # the agent transformer OUTPUT h_t is read via `agent_out_*`; the BC-loss
    # grad of h_t is pushed via `set_grad_h` before vjp and the resulting grad
    # wrt the agent input is read via `grad_agent_in_*` (feeds TaskEmbedder).
    comptime AGENT: Bool = Self.NAGENT > 0
    comptime AG_DIM: Int = Self.NAGENT * Self.D
    comptime AG_DIM_EFF: Int = _pos(Self.AG_DIM, 1)

    # Action conditioning (model.py:ActionEncoder). ACOND gates the whole
    # path; when off (ADIM=0) the action token is exactly `action_base` and
    # everything below is byte-for-byte the unconditional dynamics. ADIM_EFF
    # keeps the act-MLP type valid (IN≥1) even when unused. The MLP's second
    # layer is a ZeroLinear so the action contribution starts EXACTLY 0 ⇒ at
    # init a conditioned model equals the unconditional one (the reference
    # approximates this with fc2 std=1e-3; ZeroLinear makes it exact).
    comptime ACOND: Bool = Self.ADIM > 0
    comptime ADIM_EFF: Int = _pos(Self.ADIM, 1)
    comptime AHID_EFF: Int = _pos(Self.AHID, 2 * Self.D)
    comptime ACT_MLP = Sequential[
        Linear[Self.ADIM_EFF, Self.AHID_EFF],
        SiLU[Self.AHID_EFF],
        ZeroLinear[Self.AHID_EFF, Self.D],
    ]

    comptime PROJ = Tokenwise[Self.NSP, Linear[Self.DSP, Self.D]]
    # The transformer body (positions + block-causal stack). Split out from the
    # flow head so the FULL per-token transformer output is available: the
    # spatial columns feed the flow head, and (when AGENT) the agent columns are
    # the task output embeddings h_t. The `wm_agent_bc` dynamics stack enforces
    # agent isolation; with NAGENT=0 it is bit-identical to the old
    # `Dreamer4Stack[..., L=S, "wm_agent"]` (full mixing).
    comptime TF = Sequential[
        SinusoidalPosAddBT[Self.T, Self.S, Self.D, True],  # SCALE=True (÷√D), match refs
        Dreamer4DynStack[
            Self.D, Self.NH, Self.T, Self.NSP, Self.NREG, Self.NAGENT,
            Self.HID, Self.DEPTH, "wm_agent_bc", Self.USE_MAX,
        ],
    ]
    # The flow head: slice the spatial token outputs and apply the zero-init
    # x-prediction projection.
    comptime HEAD = Sequential[
        Slice[Self.SD, Self.SPAT_OFF, Self.REG_OFF],     # spatial tokens out
        Tokenwise[Self.NSP, ZeroLinear[Self.D, Self.DSP]],  # flow_x_head (zero)
    ]

    var proj: Self.PROJ
    var tf: Self.TF                       # positions + transformer body
    var head: Self.HEAD                   # spatial slice + flow_x_head
    var act_mlp: Self.ACT_MLP             # action encoder (used iff ACOND)
    var action_base: Param["action_base", False, Self.D]
    var signal_table: Param["signal_table", True, Self.NSIG * Self.D]
    var step_table: Param["step_table", True, Self.NSTEP * Self.D]
    var register: Param["register", False, Self.NREG * Self.D]

    # ── Working buffers (storage Tensor role) ───────────────────────────
    # Each `Tensor` owns CPU `data` (List) + optional device `dev` buffer
    # [+ pinned `hbuf` staging], lazy-grown at forward time (`ensure`/
    # `ensure_gpu`/`ensure_host`).
    var grid: Tensor           # [BATCH, S*D] transformer input grid
    var grad_grid: Tensor      # [BATCH, S*D] grad wrt grid
    var tf_out: Tensor         # [BATCH, S*D] transformer output
    var grad_tf_out: Tensor    # [BATCH, S*D] grad wrt transformer out
    var proj_out: Tensor       # [BATCH, NSP*D]
    var grad_proj_out: Tensor
    var act_out: Tensor        # [BATCH, D] act-MLP output
    var grad_act_out: Tensor   # [BATCH, D] grad into the act token
    var grad_act_in: Tensor    # [BATCH, ADIM] (discarded; data input)
    var agent_out: Tensor      # [BATCH, AG_DIM] h_t (task embeddings)
    var grad_agent_in: Tensor  # [BATCH, AG_DIM] grad wrt agent input
    # Integer index caches (CPU-canonical, set via set_indices; uploaded to
    # the staged `sig_stage`/`step_stage` device buffers on the GPU path).
    var cache_sig: List[Int]              # [BATCH] signal index per sample
    var cache_step: List[Int]             # [BATCH] step index per sample
    var sig_stage: Tensor      # [BATCH] uploaded sig indices (dev + hbuf)
    var step_stage: Tensor     # [BATCH] uploaded step indices (dev + hbuf)
    # Staged control I/O: the CPU path reads `.data`, the GPU path uploads
    # `.data`→`.hbuf`→`.dev`.
    var cache_act: Tensor          # [BATCH, ADIM] clamp/masked acts
    var cache_agent_in: Tensor     # [BATCH, AG_DIM] agent input
    var grad_agent_out: Tensor     # [BATCH, AG_DIM] h_t grad

    def __init__(out self):
        self.proj = Self.PROJ()
        self.tf = Self.TF()
        self.head = Self.HEAD()
        self.act_mlp = Self.ACT_MLP()
        self.action_base = Param["action_base", False, Self.D]()
        self.signal_table = Param["signal_table", True, Self.NSIG * Self.D]()
        self.step_table = Param["step_table", True, Self.NSTEP * Self.D]()
        self.register = Param["register", False, Self.NREG * Self.D]()
        self.grid = Tensor()
        self.grad_grid = Tensor()
        self.tf_out = Tensor()
        self.grad_tf_out = Tensor()
        self.proj_out = Tensor()
        self.grad_proj_out = Tensor()
        self.act_out = Tensor()
        self.grad_act_out = Tensor()
        self.grad_act_in = Tensor()
        self.agent_out = Tensor()
        self.grad_agent_in = Tensor()
        self.cache_sig = List[Int]()
        self.cache_step = List[Int]()
        self.sig_stage = Tensor()
        self.step_stage = Tensor()
        self.cache_act = Tensor()
        self.cache_agent_in = Tensor()
        self.grad_agent_out = Tensor()

    def _ensure_scratch_gpu(mut self, batch: Int, ctx: DeviceContext) raises:
        # Per-Tensor lazy-grow: each tracks its own device capacity. Staging
        # tensors (sig/step/act/agent-in/grad-h) also get a pinned host buffer
        # (`ensure_host`) for the H2D fill.
        self.grid.ensure_gpu(ctx, batch * Self.SD)
        self.grad_grid.ensure_gpu(ctx, batch * Self.SD)
        self.tf_out.ensure_gpu(ctx, batch * Self.SD)
        self.grad_tf_out.ensure_gpu(ctx, batch * Self.SD)
        self.proj_out.ensure_gpu(ctx, batch * Self.NSP * Self.D)
        self.grad_proj_out.ensure_gpu(ctx, batch * Self.NSP * Self.D)
        self.sig_stage.ensure_gpu(ctx, batch)
        self.sig_stage.ensure_host(ctx, batch)
        self.step_stage.ensure_gpu(ctx, batch)
        self.step_stage.ensure_host(ctx, batch)
        comptime if Self.ACOND:
            self.cache_act.ensure_gpu(ctx, batch * Self.ADIM)
            self.cache_act.ensure_host(ctx, batch * Self.ADIM)
            self.act_out.ensure_gpu(ctx, batch * Self.D)
            self.grad_act_out.ensure_gpu(ctx, batch * Self.D)
            self.grad_act_in.ensure_gpu(ctx, batch * Self.ADIM)
        comptime if Self.AGENT:
            self.cache_agent_in.ensure_gpu(ctx, batch * Self.AG_DIM)
            self.cache_agent_in.ensure_host(ctx, batch * Self.AG_DIM)
            self.agent_out.ensure_gpu(ctx, batch * Self.AG_DIM)
            self.grad_agent_out.ensure_gpu(ctx, batch * Self.AG_DIM)
            self.grad_agent_out.ensure_host(ctx, batch * Self.AG_DIM)
            self.grad_agent_in.ensure_gpu(ctx, batch * Self.AG_DIM)
        ctx.synchronize()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Dreamer4Dynamics: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.proj = Self.PROJ.make[target=target, INIT=INIT](ctx)
        # TF then HEAD consume the host RNG in the SAME order as the old fused
        # TAIL = Sequential[Pos, Stack, Slice, FlowHead], so params are
        # bit-identical to the pre-split dynamics.
        m.tf = Self.TF.make[target=target, INIT=INIT](ctx)
        m.head = Self.HEAD.make[target=target, INIT=INIT](ctx)
        m.act_mlp = Self.ACT_MLP.make[target=target, INIT=INIT](ctx)
        comptime NS = Self.NSIG * Self.D
        comptime NT = Self.NSTEP * Self.D
        comptime NR = Self.NREG * Self.D
        # Conditioning params: allocate via the storage Param factory then init
        # with the graph initializer (the reference uses normal(std=0.02) for
        # base/register; INIT is close enough — the zero-init flow head is the
        # only init that matters for convergence). Storage `init_weight[target]`
        # host-fills AND uploads to device on GPU, so one path serves both.
        m.action_base = Param["action_base", False, Self.D].make[target](ctx)
        m.signal_table = Param["signal_table", True, NS].make[target](ctx)
        m.step_table = Param["step_table", True, NT].make[target](ctx)
        m.register = Param["register", False, NR].make[target](ctx)
        INIT.init_weight[target](m.action_base.val, Self.D, 1, Self.D, ctx)
        INIT.init_weight[target](m.signal_table.val, NS, Self.NSIG, Self.D, ctx)
        INIT.init_weight[target](m.step_table.val, NT, Self.NSTEP, Self.D, ctx)
        INIT.init_weight[target](m.register.val, NR, Self.NREG, Self.D, ctx)
        return m^

    @staticmethod
    def display_label() -> String:
        return String("Dreamer4Dynamics")

    def set_indices(
        mut self,
        sig: Pointer[Scalar[DT], MutAnyOrigin],
        step: Pointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        """Push the per-sample signal/step indices (fp buffers holding exact
        non-negative ints) used by the next forward/vjp. sig ∈ [0, KMAX],
        step ∈ [0, NSTEP)."""
        if len(self.cache_sig) < batch:
            self.cache_sig.resize(batch, 0)
            self.cache_step.resize(batch, 0)
        for bt in range(batch):
            self.cache_sig[bt] = Int(Float64(sig[unsafe_offset=bt]) + 0.5)
            self.cache_step[bt] = Int(Float64(step[unsafe_offset=bt]) + 0.5)

    def set_actions(
        mut self,
        actions: Pointer[Scalar[DT], MutAnyOrigin],
        act_mask: Pointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        """Push per-sample actions for the next forward/vjp (ACOND only;
        no-op otherwise). Caches the reference's preprocessed input
        `clamp(act_mask ⊙ a, -1, 1)` (model.py:ActionEncoder.forward) so the
        act-MLP sees data only — the clamp/mask carry no param gradient. Per
        the loss contract, the MAIN forward (last) leaves `cache_act` holding
        its actions, so the act-MLP's vjp reads the correct input.

        `actions` is [batch, ADIM] row-major; `act_mask` is [ADIM] (pass all
        ones for no masking)."""
        comptime if Self.ACOND:
            self.cache_act.ensure(batch * Self.ADIM)
            for bt in range(batch):
                for a in range(Self.ADIM):
                    var v = actions[unsafe_offset=bt * Self.ADIM + a] * act_mask[unsafe_offset=a]
                    if v > Scalar[DT](1.0):
                        v = Scalar[DT](1.0)
                    elif v < Scalar[DT](-1.0):
                        v = Scalar[DT](-1.0)
                    self.cache_act.data[bt * Self.ADIM + a] = v

    # ── Agent-token control inputs / outputs (AGENT only) ───────────────
    def set_agent_in(
        mut self,
        agent_in: Pointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        """Push the per-sample agent-token INPUT for the next forward/vjp
        (AGENT only; no-op otherwise). This is the TaskEmbedder output
        `task_emb + agent_base`, a real differentiable input — unlike actions
        it carries a gradient back (read via `grad_agent_in_*` after vjp).
        `agent_in` is [batch, AG_DIM] row-major (AG_DIM = NAGENT·D)."""
        comptime if Self.AGENT:
            self.cache_agent_in.ensure(batch * Self.AG_DIM)
            for i in range(batch * Self.AG_DIM):
                self.cache_agent_in.data[i] = agent_in[unsafe_offset=i]

    def set_grad_h(
        mut self,
        grad_h: Pointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        """Push the gradient of the loss wrt the agent output h_t before vjp
        (AGENT only). Scattered into the agent columns of the transformer-out
        grad alongside the flow-head grad. `grad_h` is [batch, AG_DIM]."""
        comptime if Self.AGENT:
            self.grad_agent_out.ensure(batch * Self.AG_DIM)
            for i in range(batch * Self.AG_DIM):
                self.grad_agent_out.data[i] = grad_h[unsafe_offset=i]

    def agent_out_ptr_cpu(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """CPU pointer to h_t (the agent transformer outputs), valid after a
        CPU forward. Shape [BATCH, AG_DIM]. Sanctioned host control-output
        boundary (consumed by the BC loss / imagination rollout)."""
        return _mao(self.agent_out.data.unsafe_ptr())

    def agent_out_dev(self) -> DeviceBuffer[DT]:
        """Device buffer holding h_t, valid after a GPU forward."""
        return self.agent_out.dev.value()

    def grad_agent_in_ptr_cpu(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        """CPU pointer to the grad wrt the agent input, valid after a CPU vjp.
        Feeds the TaskEmbedder backward. Shape [BATCH, AG_DIM]."""
        return _mao(self.grad_agent_in.data.unsafe_ptr())

    def grad_agent_in_dev(self) -> DeviceBuffer[DT]:
        """Device buffer holding the grad wrt the agent input, after a GPU vjp."""
        return self.grad_agent_in.dev.value()

    def sync_agent_out(mut self, c: DeviceContext) raises:
        """D2H the agent tokens h_t (after a GPU forward) so `agent_out_ptr_cpu`
        is valid for the CPU heads. Uses the Tensor's logical-size download (the
        raw device buffer may over-allocate under lazy-grow)."""
        comptime if Self.AGENT:
            self.agent_out.download(c)

    def sync_grad_agent_in(mut self, c: DeviceContext) raises:
        """D2H the agent-input grad (after a GPU vjp) so `grad_agent_in_ptr_cpu`
        is valid for the TaskEmbedder backward."""
        comptime if Self.AGENT:
            self.grad_agent_in.download(c)

    def forward[
        target: StaticString,
        BATCH: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut output: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self.grid.ensure(BATCH * Self.SD)
            self.proj_out.ensure(BATCH * Self.NSP * Self.D)
            # proj on the module input → proj_out
            self.proj.forward[target, BATCH, POLICY=POLICY](
                inputs, self.proj_out, None
            )
            # action conditioning: act token = action_base + act_mlp(actions)
            comptime if Self.ACOND:
                self.act_out.ensure(BATCH * Self.D)
                self.act_mlp.forward[target, BATCH, POLICY=POLICY](
                    TensorRefs[1](self.cache_act), self.act_out, None
                )

            var po = TileTensor(
                self.proj_out.data, row_major[BATCH, Self.NSP * Self.D]()
            )
            var grid = TileTensor(
                self.grid.data, row_major[BATCH, Self.SD]()
            )
            var ab = TileTensor(self.action_base.val.data, row_major[Self.D]())
            var sigt = TileTensor(
                self.signal_table.val.data, row_major[Self.NSIG * Self.D]()
            )
            var stpt = TileTensor(
                self.step_table.val.data, row_major[Self.NSTEP * Self.D]()
            )
            var reg = TileTensor(
                self.register.val.data, row_major[Self.NREG * Self.D]()
            )
            for bt in range(BATCH):
                var si = self.cache_sig[bt]   # set via set_indices()
                var pi = self.cache_step[bt]
                for d in range(Self.D):
                    grid[bt, d] = ab[d]                          # action
                    comptime if Self.ACOND:
                        grid[bt, d] += self.act_out.data[bt * Self.D + d]
                    grid[bt, Self.D + d] = sigt[si * Self.D + d]  # signal
                    grid[bt, 2 * Self.D + d] = stpt[pi * Self.D + d]  # step
                for k in range(Self.NSP * Self.D):
                    grid[bt, Self.SPAT_OFF + k] = po[bt, k]       # spatial
                for k in range(Self.NREG * Self.D):
                    grid[bt, Self.REG_OFF + k] = reg[k]           # register
                comptime if Self.AGENT:
                    for k in range(Self.AG_DIM):
                        grid[bt, Self.AGENT_OFF + k] = self.cache_agent_in.data[
                            bt * Self.AG_DIM + k
                        ]

            # transformer body → full per-token output
            self.tf_out.ensure(BATCH * Self.SD)
            self.tf.forward[target, BATCH, POLICY=POLICY](
                TensorRefs[1](self.grid), self.tf_out, None
            )
            # agent token outputs h_t = transformer output agent columns
            comptime if Self.AGENT:
                self.agent_out.ensure(BATCH * Self.AG_DIM)
                for bt in range(BATCH):
                    for k in range(Self.AG_DIM):
                        self.agent_out.data[bt * Self.AG_DIM + k] = (
                            self.tf_out.data[
                                bt * Self.SD + Self.AGENT_OFF + k
                            ]
                        )
            # flow head reads the spatial columns of the transformer output
            self.head.forward[target, BATCH, POLICY=POLICY](
                TensorRefs[1](self.tf_out), output, None
            )
        else:
            var c = ctx.value()
            self._ensure_scratch_gpu(BATCH, c)
            # upload the cached indices (set via set_indices) → device
            var sh = self.sig_stage.hbuf.value()
            var th = self.step_stage.hbuf.value()
            for bt in range(BATCH):
                sh.unsafe_ptr()[unsafe_offset=bt] = Scalar[DT](Float64(self.cache_sig[bt]))
                th.unsafe_ptr()[unsafe_offset=bt] = Scalar[DT](Float64(self.cache_step[bt]))
            c.enqueue_copy(self.sig_stage.dev.value(), sh)
            c.enqueue_copy(self.step_stage.dev.value(), th)
            comptime if Self.ACOND:
                var ah = self.cache_act.hbuf.value()
                for i in range(BATCH * Self.ADIM):
                    ah.unsafe_ptr()[unsafe_offset=i] = self.cache_act.data[i]
                c.enqueue_copy(self.cache_act.dev.value(), ah)
            comptime if Self.AGENT:
                var agh = self.cache_agent_in.hbuf.value()
                for i in range(BATCH * Self.AG_DIM):
                    agh.unsafe_ptr()[unsafe_offset=i] = self.cache_agent_in.data[i]
                c.enqueue_copy(self.cache_agent_in.dev.value(), agh)

            self.proj_out.ensure_gpu(c, BATCH * Self.NSP * Self.D)
            self.proj.forward[target, BATCH, POLICY=POLICY](
                inputs, self.proj_out, ctx
            )

            comptime AN = BATCH * Self.SD
            comptime PN = BATCH * Self.NSP * Self.D
            var proj_lt = tile[PN](self.proj_out.dev.value())
            var ab_lt = tile[Self.D](self.action_base.val.dev.value())
            var sg_lt = tile[Self.NSIG * Self.D](self.signal_table.val.dev.value())
            var st_lt = tile[Self.NSTEP * Self.D](self.step_table.val.dev.value())
            var rg_lt = tile[Self.NREG * Self.D](self.register.val.dev.value())
            var si_lt = tile[BATCH](self.sig_stage.dev.value())
            var sp_lt = tile[BATCH](self.step_stage.dev.value())
            var grid_lt = tile[AN](self.grid.dev.value())
            comptime ak = _dyn_assemble_kernel[
                BATCH, Self.S, Self.D, Self.NSP, Self.NREG, Self.NSIG, Self.NSTEP
            ]
            c.enqueue_function[ak](
                proj_lt, ab_lt, sg_lt, st_lt, rg_lt, si_lt, sp_lt, grid_lt,
                grid_dim=(AN + TPB - 1) // TPB, block_dim=TPB,
            )
            # action conditioning: act token += act_mlp(actions)
            comptime if Self.ACOND:
                self.act_out.ensure_gpu(c, BATCH * Self.D)
                self.act_mlp.forward[target, BATCH, POLICY=POLICY](
                    TensorRefs[1](self.cache_act), self.act_out, ctx
                )
                var aout_lt = tile[BATCH * Self.D](self.act_out.dev.value())
                comptime addk = _dyn_add_act_kernel[BATCH, Self.S, Self.D]
                c.enqueue_function[addk](
                    aout_lt, grid_lt,
                    grid_dim=(BATCH * Self.D + TPB - 1) // TPB, block_dim=TPB,
                )
            # agent token input → grid agent columns
            comptime if Self.AGENT:
                var agin_lt = tile[BATCH * Self.AG_DIM](self.cache_agent_in.dev.value())
                comptime stk = _dyn_set_agent_kernel[
                    BATCH, Self.S, Self.D, Self.NAGENT
                ]
                c.enqueue_function[stk](
                    agin_lt, grid_lt,
                    grid_dim=(BATCH * Self.AG_DIM + TPB - 1) // TPB,
                    block_dim=TPB,
                )

            self.tf_out.ensure_gpu(c, BATCH * Self.SD)
            self.tf.forward[target, BATCH, POLICY=POLICY](
                TensorRefs[1](self.grid), self.tf_out, ctx
            )
            # h_t = transformer output agent columns
            comptime if Self.AGENT:
                var tfout_lt = tile[BATCH * Self.SD](self.tf_out.dev.value())
                var agout_lt = tile[BATCH * Self.AG_DIM](self.agent_out.dev.value())
                comptime exk = _dyn_extract_agent_fwd_kernel[
                    BATCH, Self.S, Self.D, Self.NAGENT
                ]
                c.enqueue_function[exk](
                    tfout_lt, agout_lt,
                    grid_dim=(BATCH * Self.AG_DIM + TPB - 1) // TPB,
                    block_dim=TPB,
                )
            self.head.forward[target, BATCH, POLICY=POLICY](
                TensorRefs[1](self.tf_out), output, ctx
            )

    def vjp[
        target: StaticString,
        BATCH: Int,
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
        comptime if target == "cpu":
            self.grad_grid.ensure(BATCH * Self.SD)
            self.grad_tf_out.ensure(BATCH * Self.SD)
            self.grad_proj_out.ensure(BATCH * Self.NSP * Self.D)
            # flow head backward → grad wrt the transformer output. The Slice
            # vjp zero-fills then scatters, so grad_tf_out has the spatial
            # columns filled and ALL other columns (incl. agent) zero.
            self.head.vjp[target, BATCH, POLICY=POLICY](
                TensorRefs[1](self.tf_out), grad_output,
                TensorRefs[1](self.grad_tf_out), ctx,
            )
            # add the h_t grad (set via set_grad_h) into the agent columns
            comptime if Self.AGENT:
                for bt in range(BATCH):
                    for k in range(Self.AG_DIM):
                        self.grad_tf_out.data[
                            bt * Self.SD + Self.AGENT_OFF + k
                        ] += self.grad_agent_out.data[bt * Self.AG_DIM + k]
            # transformer backward → grad wrt the assembled grid
            self.tf.vjp[target, BATCH, POLICY=POLICY](
                TensorRefs[1](self.grid), self.grad_tf_out,
                TensorRefs[1](self.grad_grid), ctx,
            )

            # spatial token grad → proj input grad
            var ggrid = TileTensor(
                self.grad_grid.data, row_major[BATCH, Self.SD]()
            )
            var gpo = TileTensor(
                self.grad_proj_out.data, row_major[BATCH, Self.NSP * Self.D]()
            )
            for bt in range(BATCH):
                for k in range(Self.NSP * Self.D):
                    gpo[bt, k] = ggrid[bt, Self.SPAT_OFF + k]
            self.proj.vjp[target, BATCH, POLICY=POLICY](
                forward_input, self.grad_proj_out, grad_inputs, ctx
            )

            # agent input grad = grid grad agent columns (feeds the TaskEmbedder
            # backward).
            comptime if Self.AGENT:
                self.grad_agent_in.ensure(BATCH * Self.AG_DIM)
                for bt in range(BATCH):
                    for k in range(Self.AG_DIM):
                        self.grad_agent_in.data[bt * Self.AG_DIM + k] = ggrid[
                            bt, Self.AGENT_OFF + k
                        ]

            # raw conditioning-param grads (storage vjp always computes them —
            # the legacy mode="all" path is now unconditional).
            var gab = TileTensor(self.action_base.grd.data, row_major[Self.D]())
            var gsig = TileTensor(
                self.signal_table.grd.data, row_major[Self.NSIG * Self.D]()
            )
            var gstp = TileTensor(
                self.step_table.grd.data, row_major[Self.NSTEP * Self.D]()
            )
            var greg = TileTensor(
                self.register.grd.data, row_major[Self.NREG * Self.D]()
            )
            for bt in range(BATCH):
                var si = self.cache_sig[bt]
                var pi = self.cache_step[bt]
                for d in range(Self.D):
                    gab[d] += ggrid[bt, d]
                    gsig[si * Self.D + d] += ggrid[bt, Self.D + d]
                    gstp[pi * Self.D + d] += ggrid[bt, 2 * Self.D + d]
                for k in range(Self.NREG * Self.D):
                    greg[k] += ggrid[bt, Self.REG_OFF + k]

            # action MLP: token = action_base + act_mlp(actions), so the act
            # token's grad (ggrid[:, 0:D]) — the same grad accumulated into
            # action_base above — is the act-MLP's grad_output. Its grad_input
            # (wrt actions) is data ⇒ discarded.
            comptime if Self.ACOND:
                self.grad_act_out.ensure(BATCH * Self.D)
                self.grad_act_in.ensure(BATCH * Self.ADIM)
                for bt in range(BATCH):
                    for d in range(Self.D):
                        self.grad_act_out.data[bt * Self.D + d] = ggrid[bt, d]
                self.act_mlp.vjp[target, BATCH, POLICY=POLICY](
                    TensorRefs[1](self.cache_act), self.grad_act_out,
                    TensorRefs[1](self.grad_act_in), ctx,
                )
        else:
            var c = ctx.value()
            self._ensure_scratch_gpu(BATCH, c)
            comptime AN = BATCH * Self.SD
            comptime PN = BATCH * Self.NSP * Self.D
            # flow head backward → grad wrt transformer output (Slice vjp
            # zero-fills, so agent/other columns are 0)
            self.head.vjp[target, BATCH, POLICY=POLICY](
                TensorRefs[1](self.tf_out), grad_output,
                TensorRefs[1](self.grad_tf_out), ctx,
            )
            comptime if Self.AGENT:
                # upload h_t grad (set via set_grad_h), add into agent columns
                var goh2 = self.grad_agent_out.hbuf.value()
                for i in range(BATCH * Self.AG_DIM):
                    goh2.unsafe_ptr()[unsafe_offset=i] = self.grad_agent_out.data[i]
                c.enqueue_copy(self.grad_agent_out.dev.value(), goh2)
                var gagout_lt = tile[BATCH * Self.AG_DIM](self.grad_agent_out.dev.value())
                var gtfout_lt = tile[BATCH * Self.SD](self.grad_tf_out.dev.value())
                comptime agk = _dyn_add_agent_grad_kernel[
                    BATCH, Self.S, Self.D, Self.NAGENT
                ]
                c.enqueue_function[agk](
                    gagout_lt, gtfout_lt,
                    grid_dim=(BATCH * Self.AG_DIM + TPB - 1) // TPB,
                    block_dim=TPB,
                )
            # transformer backward → grad wrt the assembled grid
            self.tf.vjp[target, BATCH, POLICY=POLICY](
                TensorRefs[1](self.grid), self.grad_tf_out,
                TensorRefs[1](self.grad_grid), ctx,
            )

            var ggrid_lt = tile[AN](self.grad_grid.dev.value())
            var gpo_lt = tile[PN](self.grad_proj_out.dev.value())
            comptime gpk = _dyn_grad_proj_kernel[BATCH, Self.S, Self.D, Self.NSP]
            c.enqueue_function[gpk](
                ggrid_lt, gpo_lt, grid_dim=(PN + TPB - 1) // TPB, block_dim=TPB,
            )
            self.proj.vjp[target, BATCH, POLICY=POLICY](
                forward_input, self.grad_proj_out, grad_inputs, ctx
            )

            # agent input grad = grid grad agent columns (feeds the TaskEmbedder
            # backward).
            comptime if Self.AGENT:
                var gagin_lt = tile[BATCH * Self.AG_DIM](self.grad_agent_in.dev.value())
                comptime egk = _dyn_extract_agent_grad_kernel[
                    BATCH, Self.S, Self.D, Self.NAGENT
                ]
                c.enqueue_function[egk](
                    ggrid_lt, gagin_lt,
                    grid_dim=(BATCH * Self.AG_DIM + TPB - 1) // TPB,
                    block_dim=TPB,
                )

            # raw conditioning-param grads (storage vjp always computes them).
            var gab_lt = tile[Self.D](self.action_base.grd.dev.value())
            var greg_lt = tile[Self.NREG * Self.D](self.register.grd.dev.value())
            var gsig_lt = tile[Self.NSIG * Self.D](self.signal_table.grd.dev.value())
            var gstp_lt = tile[Self.NSTEP * Self.D](self.step_table.grd.dev.value())
            var si_lt = tile[BATCH](self.sig_stage.dev.value())
            var sp_lt = tile[BATCH](self.step_stage.dev.value())
            comptime bk = _dyn_grad_base_kernel[BATCH, Self.S, Self.D]
            c.enqueue_function[bk](
                ggrid_lt, gab_lt,
                grid_dim=(Self.D + TPB - 1) // TPB, block_dim=TPB,
            )
            comptime rk = _dyn_grad_reg_kernel[
                BATCH, Self.S, Self.D, Self.NSP, Self.NREG
            ]
            c.enqueue_function[rk](
                ggrid_lt, greg_lt,
                grid_dim=(Self.NREG * Self.D + TPB - 1) // TPB,
                block_dim=TPB,
            )
            comptime sk = _dyn_grad_sig_kernel[
                BATCH, Self.S, Self.D, Self.NSIG
            ]
            c.enqueue_function[sk](
                ggrid_lt, si_lt, gsig_lt,
                grid_dim=(Self.NSIG * Self.D + TPB - 1) // TPB,
                block_dim=TPB,
            )
            comptime tk = _dyn_grad_step_kernel[
                BATCH, Self.S, Self.D, Self.NSTEP
            ]
            c.enqueue_function[tk](
                ggrid_lt, sp_lt, gstp_lt,
                grid_dim=(Self.NSTEP * Self.D + TPB - 1) // TPB,
                block_dim=TPB,
            )
            # action MLP grad: act-token grad (ggrid col [0,D)) → act_mlp.vjp
            comptime if Self.ACOND:
                var gaout_lt = tile[BATCH * Self.D](self.grad_act_out.dev.value())
                comptime xk = _dyn_extract_token0_kernel[
                    BATCH, Self.S, Self.D
                ]
                c.enqueue_function[xk](
                    ggrid_lt, gaout_lt,
                    grid_dim=(BATCH * Self.D + TPB - 1) // TPB,
                    block_dim=TPB,
                )
                self.act_mlp.vjp[target, BATCH, POLICY=POLICY](
                    TensorRefs[1](self.cache_act), self.grad_act_out,
                    TensorRefs[1](self.grad_act_in), ctx,
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        # raw conditioning params (action_base, signal/step tables, register)
        for_each_param_auto[Self, V, target](self, visitor, ctx, prefix)
        # child modules
        self.proj.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "proj")
        )
        self.tf.for_each_param[target, V](visitor, ctx, join_name(prefix, "tf"))
        self.head.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "head")
        )
        comptime if Self.ACOND:
            self.act_mlp.for_each_param[target, V](
                visitor, ctx, join_name(prefix, "act_mlp")
            )

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        zero_grad_auto[Self, target](self, ctx)
        self.proj.zero_grad[target](ctx)
        self.tf.zero_grad[target](ctx)
        self.head.zero_grad[target](ctx)
        comptime if Self.ACOND:
            self.act_mlp.zero_grad[target](ctx)
