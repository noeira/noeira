"""TiedLinear[IN, OUT] — a bias-less Linear whose weight is *borrowed* from
another leaf (weight tying, nanoGPT's `lm_head.weight = wte.weight`).

Transformed from legacy `nn.primitives.TiedLinear` onto the storage surface.
The canonical use is the GPT LM head sharing the token embedding's table: the
embedding owns a `[VOCAB, EMBED]` weight; the LM head is the same map applied
transposed (`out = x @ Wᵀ`). PyTorch ties by making both modules reference one
`nn.Parameter`; storage leaves own private `Param` storage, so instead
`TiedLinear`:

  * owns **no** weight `Param` (and no bias) — so reflection
    (`for_each_param`) finds nothing here (no `IsParam` field) and the
    optimizer / checkpoint walk collects the shared weight EXACTLY ONCE, via
    the *source* leaf (the embedding). `for_each_param`/`zero_grad` therefore
    inherit the Module reflection defaults, which reflect to a no-op here;
  * holds a **borrowed reference to the source weight's storage cells**
    (`src_val` / `src_grd`), wired once after construction via `tie_to`
    (the analog of `lm_head.weight = wte.weight` in __init__);
  * reads the source value transposed in forward / grad-input, and
    **accumulates** its weight gradient straight into the source grad
    `Tensor` (`+=`) in `vjp`. Both leaves accumulating into one grad cell is
    correct because storage grads are `+=` and `zero_grad` clears the cell
    once per step (the source leaf owns + zeroes it).

BORROWING REPRESENTATION (the crux): the borrow is the SAFE `std.memory.Pointer`
(single-element, no arithmetic, bounds-safe deref — NOT a raw `Pointer`),
holding a `Pointer[Tensor, MutAnyOrigin]` to the owner's `Param.val` /
`Param.grd` *storage cells*. Dereferencing yields a mutable `Tensor`, so
forward/vjp build their typed views with the SAME storage-surface calls every
other leaf uses — `TileTensor(t.data, …)` / cblas on CPU, `t.lt["gpu", layout]()`
on GPU. `tie_to` takes the source cells BY `ref` and builds the `Pointer`
internally, so no pointer is constructed at the call site.

The origin is the wildcard `MutAnyOrigin` (not a TRACKED origin `o` like
`TensorRefs`) DELIBERATELY: a tracked `Pointer[Tensor, o]` would force a
borrow-origin TYPE PARAMETER onto this struct, which would ripple into every
combinator that holds it (`Sequential`/`Tokenwise`/… are parametric over
`*MODULES: Module` and would each have to thread `o`). The wildcard keeps the
struct origin-free so it conforms to `Module` and drops into a generic
`Sequential` trained by the standard `Trainer` loop. The trade: the wildcard
does not pin the owner, so the POINTER-STABILITY RULE below is load-bearing.
This is the storage-clean evolution of the legacy raw `Pointer[Scalar]`
to the val/grad buffers: safe single-element reference, Tensor-cell granularity.

bf16-FLOW (AMP "Step B"): `TiedLinear[IN, OUT]` is fp32 (unchanged), while
`TiedLinear[IN, OUT, DType.bfloat16]` flows its ACTIVATIONS at bf16
(`ACT_DT == bfloat16`). Like Linear, the bf16 GEMMs run on a bf16 weight while
the master weight/grad (BORROWED from the source leaf) stay fp32. The crux vs
Linear: the tied weight CHANGES every optimizer step (it is the embedding's live
weight, updated each step), so the bf16 weight cast is NOT version-gated — the
borrowed weight is recast → bf16 EVERY forward (`w_bf`, [OUT, IN]) and the bf16
cast reused in vjp (no optimizer step between a fwd and its bwd). grad_W
accumulates bf16→fp32 into the borrowed source grad cell (unchanged tie
mechanism). The fp32 (ACT_DT == DT) path is byte-for-byte the legacy NoAMP path;
the bf16 path is GPU-only.

POINTER-STABILITY RULE (load-bearing): `tie_to` captures a wildcard-origin
reference to the owner's `Param.val` / `Param.grd` *cells*, so the OWNER must
outlive every forward/vjp on this head. In real use that is automatic — the
source leaf (embedding) lives inside the same model struct that owns this head
— but it means `tie_to` must run after the owning model reaches its final home
(and after any load), and a standalone owner cell (e.g. in a test) must be kept
alive past the tied calls. A destroyed owner leaves a dangling reference; on GPU
that surfaces as the head reading a freed device buffer (zeros / empty `.dev`).

Shapes (IN = EMBED, OUT = VOCAB; source weight is `[OUT, IN]`):
  * forward:        out[b,v]  = Σ_e x[b,e]·Wsrc[v,e]      (= x @ Wsrcᵀ)
  * vjp grad-input: dx[b,e]   = Σ_v dout[b,v]·Wsrc[v,e]   (= dout @ Wsrc)
  * vjp grad-weight: dWsrc[v,e] += Σ_b dout[b,v]·x[b,e]   (accumulate)
"""

from std.sys import CompilationTarget
from std.memory import Pointer
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from .linear import _cast_f2b_kernel


# ──────────────────────────────────────────────────────────────────────
# GPU param-grad kernel — accumulate dWsrc[v,e] += Σ_b dout[b,v]·x[b,e]
# into the borrowed source grad buffer. One thread per (v, e); serial over
# BATCH. Carried over verbatim from legacy (mirrors Embedding's
# `_embedding_grad_w_kernel`, so the two accumulate symmetrically into the
# shared buffer). Dtype-parametric (`ADT`) on the grad_output + cache_in
# activations: the bf16 path reads bf16 `grad_output`/`cache_in` and
# accumulates into the FP32 `grad_w` (each product cast to DT — the accumulator
# stays fp32 master).
# ──────────────────────────────────────────────────────────────────────


def _tied_grad_w_kernel[
    BATCH: Int, OUT: Int, IN: Int, ADT: DType = DT
](
    grad_output: Tile[ADT, BATCH, OUT],
    cache_in: Tile[ADT, BATCH, IN],
    grad_w: Tile[DT, OUT, IN],
):
    var gid = Int(global_idx.x)
    comptime total = OUT * IN
    if gid >= total:
        return
    var v = gid // IN
    var e = gid % IN
    var acc: Scalar[DT] = 0.0
    for b in range(BATCH):
        acc += rebind[Scalar[ADT]](grad_output[b, v]).cast[DT]() * rebind[
            Scalar[ADT]
        ](cache_in[b, e]).cast[DT]()
    grad_w[v, e] = rebind[Scalar[DT]](grad_w[v, e]) + acc


struct TiedLinear[IN_: Int, OUT_: Int, ADT: DType = DT](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_
    # Activation-flow dtype (satisfies the Module trait). `TiedLinear[IN, OUT]`
    # = fp32 (ACT_DT == DT, the legacy path); `TiedLinear[IN, OUT, bfloat16]`
    # flows activations at bf16 (the AMP "Step B" memory win).
    comptime ACT_DT = Self.ADT

    # No owned weight/bias Param — the weight is borrowed (reflection finds no
    # `IsParam` field here, so the optimizer never double-counts the shared
    # weight). `tie_to` points these at the source weight's val/grad cells.
    # SAFE `Pointer` (not raw `Pointer`); wildcard origin so the struct
    # stays origin-free (see module docstring).
    var src_val: Optional[Pointer[Tensor, MutUntrackedOrigin]]
    var src_grd: Optional[Pointer[Tensor, MutUntrackedOrigin]]
    # bf16-flow compute scratch (lazy; used only when ACT_DT == bf16 and
    # target == "gpu"). `w_bf` [OUT, IN] is the bf16 copy of the BORROWED source
    # weight. Unlike Linear, this is NOT version-gated: the tied weight changes
    # every optimizer step (it is the source leaf's live weight), so it is
    # recast → bf16 EVERY forward (and reused in vjp — no opt step between a fwd
    # and its bwd).
    var w_bf: TensorImpl[Self.ADT]

    def __init__(out self):
        self.src_val = None
        self.src_grd = None
        self.w_bf = TensorImpl[Self.ADT]()

    # ----- Factory (INIT unused — no owned params) ------------------------

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TiedLinear: target must be 'cpu' or 'gpu'"
        return Self()

    # ----- Tie wiring -----------------------------------------------------

    def tie_to(mut self, ref val: Tensor, ref grd: Tensor):
        """Point this head at the source weight's value + grad storage cells
        (the owner's `Param.val` / `Param.grd`, laid out `[OUT, IN]`). Pass the
        cells BY `ref` (e.g. `head.tie_to(emb.weight.val, emb.weight.grd)`);
        the safe `Pointer` is built internally — no pointer at the call site.
        Call once after the owning model settles in its final home (and after
        any load); the source leaf keeps the cells alive. Idempotent. The
        captured origin is the wildcard `MutAnyOrigin` (see module docstring),
        so the POINTER-STABILITY RULE applies: the owner must outlive this head.
        """
        self.src_val = rebind[Pointer[Tensor, MutUntrackedOrigin]](Pointer(to=val))
        self.src_grd = rebind[Pointer[Tensor, MutUntrackedOrigin]](Pointer(to=grd))

    def tie_to_ptr(
        mut self,
        val: Pointer[Tensor, MutAnyOrigin],
        grd: Pointer[Tensor, MutAnyOrigin],
    ):
        """Wire via pre-built wildcard `Pointer` VALUES (vs `tie_to`'s `ref`
        args). Use this when the owner and this head are BOTH sub-paths of one
        enclosing struct (e.g. a GPT model where the embedding owns the weight
        and the LM head borrows it): a `ref` arg would borrow the enclosing
        struct immutably while `mut self` borrows it mutably → exclusivity
        conflict. A wildcard `Pointer` value holds NO tracked borrow of the
        enclosing struct, so the caller can build it (`rebind[...](Pointer(to=
        owner.weight.val))` into a local) — releasing the structural borrow —
        then call this. Same POINTER-STABILITY RULE applies (owner outlives
        head). Idempotent."""
        self.src_val = rebind[Pointer[Tensor, MutUntrackedOrigin]](val)
        self.src_grd = rebind[Pointer[Tensor, MutUntrackedOrigin]](grd)

    # Return the field's own origin (`MutUntrackedOrigin`). Nightly no longer
    # implicitly widens `MutUntrackedOrigin` → `MutAnyOrigin`; every caller only
    # dereferences the result, so matching the field is both accurate and
    # sufficient.
    def _val(self) raises -> Pointer[Tensor, MutUntrackedOrigin]:
        if not self.src_val:
            raise Error(
                "TiedLinear: not wired — call tie_to(...) before forward/vjp"
            )
        return self.src_val.value()

    def _grd(self) raises -> Pointer[Tensor, MutUntrackedOrigin]:
        if not self.src_grd:
            raise Error(
                "TiedLinear: not wired — call tie_to(...) before forward/vjp"
            )
        return self.src_grd.value()

    def _ensure_w_bf(mut self, c: DeviceContext) raises:
        """Recast the BORROWED source weight (`_val()`, [OUT, IN]) → the bf16
        `w_bf` EVERY call (the tie weight is always dirty — the source leaf's
        optimizer step bumps it each step, so no version gate). Called from
        forward (the cast) and reused in vjp (no opt step between a fwd/bwd)."""
        ref w = self._val()[]
        self.w_bf.ensure_gpu(c, Self.W_SIZE)
        c.enqueue_function[_cast_f2b_kernel](
            w.dev.value(),
            self.w_bf.dev.value(),
            Int64(Self.W_SIZE),
            grid_dim=(Self.W_SIZE + 255) // 256,
            block_dim=256,
        )

    # ----- Forward: out = x @ Wsrcᵀ ---------------------------------------

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            ref w = self._val()[]  # source weight cell [OUT, IN]
            comptime if target == "cpu":
                outd.ensure(B * Self.OUT_)
                var x_v = TileTensor(in0d.data, row_major[B, Self.IN_]())
                var w_v = TileTensor(w.data, row_major[Self.OUT_, Self.IN_]())
                var out_v = TileTensor(outd.data, row_major[B, Self.OUT_]())
                # out[b,v] = Σ_e x[b,e]·Wsrc[v,e]  (transpose_b: contract IN).
                max_matmul[transpose_b=True, target="cpu"](
                    out_v, x_v, w_v, None
                )
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, B * Self.OUT_)
                var x_v = TileTensor(in0d.dev.value(), row_major[B, Self.IN_]())
                var w_v = TileTensor(
                    w.dev.value(), row_major[Self.OUT_, Self.IN_]()
                )
                var out_v = TileTensor(
                    outd.dev.value(), row_major[B, Self.OUT_]()
                )
                # out[b,v] = Σ_e x[b,e]·Wsrc[v,e]  (transpose_b: contract IN).
                max_matmul[transpose_b=True, target="gpu"](out_v, x_v, w_v, c)
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow TiedLinear is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            # x (in0) is ALREADY bf16. W: recast the borrowed source weight →
            # bf16 every forward (the tie weight is always dirty).
            self._ensure_w_bf(c)
            var x_v = TileTensor(in0.dev.value(), row_major[B, Self.IN_]())
            var w_bf_v = TileTensor(
                self.w_bf.dev.value(), row_major[Self.OUT_, Self.IN_]()
            )
            var out_v = TileTensor(out.dev.value(), row_major[B, Self.OUT_]())
            # out[b,v] = Σ_e x[b,e]·Wsrc[v,e] (transpose_b; bf16-in → bf16-out).
            max_matmul[transpose_b=True, target="gpu"](out_v, x_v, w_bf_v, c)

    # ----- Backward (combined; Embedding-style single vjp) ----------------

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
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            ref find = rebind[Tensor](fin)
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            ref w = self._val()[]  # source weight cell [OUT, IN]
            ref gw = self._grd()[]  # source grad cell [OUT, IN] (accumulate)
            comptime if target == "cpu":
                gind.ensure(B * Self.IN_)
                var go_v = TileTensor(
                    god.data, row_major[B, Self.OUT_]()
                )
                var gi_v = TileTensor(gind.data, row_major[B, Self.IN_]())
                var w_v = TileTensor(w.data, row_major[Self.OUT_, Self.IN_]())
                # ── (1) grad-weight: dWsrc[v,e] += Σ_b dout[b,v]·x[b,e] =
                # doutᵀ@x. Apple-fp32: ONE fused cblas_sgemm (TRANSPOSE A=dout,
                # beta=1 → no transpose buffer, no temp, no accumulate loop).
                # Else: naive loops.
                comptime IS_APPLE_F32 = (
                    CompilationTarget.is_macos() and DT == DType.float32
                )
                comptime if IS_APPLE_F32:
                    var cblas = get_cblas_f32_function()
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.OUT_),
                        Int32(Self.IN_),
                        Int32(B),
                        Float32(1.0),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            god.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            find.data.unsafe_ptr()
                        ),
                        Int32(Self.IN_),
                        Float32(1.0),
                        rebind[Pointer[Float32, MutAnyOrigin]](
                            gw.data.unsafe_ptr()
                        ),
                        Int32(Self.IN_),
                    )
                else:
                    # Non-Apple CPU: dWsrc = doutᵀ @ x via the generic CPU GEMM
                    # (transpose dout → temp, then max_matmul + accumulate;
                    # mirrors Linear/Embedding's grad_w).
                    var goT = List[Scalar[DT]](
                        length=Self.OUT_ * B, fill=Scalar[DT](0)
                    )
                    for b in range(B):
                        for v in range(Self.OUT_):
                            goT[v * B + b] = go_v[b, v]
                    var dW_tmp = List[Scalar[DT]](
                        length=Self.OUT_ * Self.IN_, fill=Scalar[DT](0)
                    )
                    var goT_tt = TileTensor(goT, row_major[Self.OUT_, B]())
                    var x_v = TileTensor(find.data, row_major[B, Self.IN_]())
                    var dW_tt = TileTensor(
                        dW_tmp, row_major[Self.OUT_, Self.IN_]()
                    )
                    max_matmul[target="cpu"](dW_tt, goT_tt, x_v, None)
                    var gw_p = gw.data.unsafe_ptr()
                    for i in range(Self.OUT_ * Self.IN_):
                        gw_p[unsafe_offset=i] += dW_tmp[i]
                # ── (2) grad-input = dout @ Wsrc.
                max_matmul[target="cpu"](gi_v, go_v, w_v, None)
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, B * Self.IN_)
                # ── (1) grad-weight kernel (accumulate into source grad cell).
                var go_lt = god.tt["gpu", B, Self.OUT_]()
                var x_lt = find.tt["gpu", B, Self.IN_]()
                var gw_lt = gw.tt["gpu", Self.OUT_, Self.IN_]()
                comptime total = Self.W_SIZE
                comptime n_blocks = (total + TPB - 1) // TPB
                c.enqueue_function[
                    _tied_grad_w_kernel[B, Self.OUT_, Self.IN_]
                ](go_lt, x_lt, gw_lt, grid_dim=n_blocks, block_dim=TPB)
                # ── (2) grad-input = dout @ Wsrc.
                var go_v = TileTensor(
                    god.dev.value(), row_major[B, Self.OUT_]()
                )
                var w_v = TileTensor(
                    w.dev.value(), row_major[Self.OUT_, Self.IN_]()
                )
                var gi_v = TileTensor(
                    gind.dev.value(), row_major[B, Self.IN_]()
                )
                max_matmul[target="gpu"](gi_v, go_v, w_v, c)
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow TiedLinear is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_)
            ref gw = self._grd()[]  # source grad cell [OUT, IN] (fp32 master)
            # fin/grad_output ALREADY bf16. W reuses the forward's bf16 cast.
            self._ensure_w_bf(c)
            # ── (1) grad-weight: dWsrc[v,e] += Σ_b dout[b,v]·x[b,e]. bf16
            # go/cache_in → fp32 master grad (fp32 accumulator in the kernel).
            var go_lt = grad_output.tt["gpu", B, Self.OUT_]()
            var x_lt = fin.tt["gpu", B, Self.IN_]()
            var gw_lt = gw.tt["gpu", Self.OUT_, Self.IN_]()
            comptime total = Self.W_SIZE
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[
                _tied_grad_w_kernel[B, Self.OUT_, Self.IN_, Self.ADT]
            ](go_lt, x_lt, gw_lt, grid_dim=n_blocks, block_dim=TPB)
            # ── (2) grad-input = dout @ Wsrc → bf16 gin (bf16-in, bf16-out).
            var go_v = TileTensor(
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var w_bf_v = TileTensor(
                self.w_bf.dev.value(), row_major[Self.OUT_, Self.IN_]()
            )
            var gi_v = TileTensor(gin.dev.value(), row_major[B, Self.IN_]())
            max_matmul[target="gpu"](gi_v, go_v, w_bf_v, c)

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (no `IsParam` field here → reflects to a no-op; the shared weight is
    # walked exactly once, by the source leaf that owns it).
