"""ACT_DT threading spike (doc-Phase-2 de-risk).

Proves the bf16-FLOW plumbing: activations are STORED at `ACT_DT` (bf16 under AMP)
and flow between layers, with the persistent intermediate buffers a combinator
needs for vjp living as `TensorImpl[child.ACT_DT]` FIELDS. This works only because
ACT_DT is an ASSOCIATED COMPTIME of the module trait, carried as a struct param —
so a combinator's child TYPES are struct params and `child.ACT_DT` is known at
field-declaration time. (POLICY-as-method-param can't do this: a field type can't
depend on a method's comptime.)

Proof obligations (all on Apple, dtype-agnostic / NoAMP numeric):
  1. ACT_DT threads through trait + combinator + leaf and COMPILES.
  2. NoAMP (ACT_DT=fp32) is BIT-IDENTICAL to the production Sequential[Linear,Linear].
  3. bf16-flow (ACT_DT=bf16) RUNS end-to-end (intermediate activation is bf16);
     numerics are garbage on Apple (Metal bf16 linalg bug) — NVIDIA-gated.

Run: pixi run -e apple mojo run -I . tests/nn/test_amp_act_dt_spike.mojo
"""

from std.memory import Pointer
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile
from linalg.matmul import matmul as max_matmul

from std.testing import assert_true
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import Param
from noeira.nn.core.initializer import Initializer, Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential

comptime BF16 = DType.bfloat16


# ── ACT_DT-parametrized borrowing ref-pack (mirrors TensorRefs) ──────────
@fieldwise_init
struct TensorRefsA[N: Int, o: MutOrigin, ADT: DType = DT](Copyable, Movable):
    var ptrs: Array[Pointer[TensorImpl[Self.ADT], Self.o], Self.N]

    def __init__(out self, ref[Self.o] t: TensorImpl[Self.ADT]) raises:
        comptime assert Self.N == 1, "spike: N==1 only"
        self.ptrs = Array[Pointer[TensorImpl[Self.ADT], Self.o], Self.N](
            fill=Pointer(to=t)
        )

    def __getitem__(self, i: Int) -> ref[Self.o] TensorImpl[Self.ADT]:
        return self.ptrs[i][]


# ── module trait with ACT_DT as an ASSOCIATED COMPTIME ───────────────────
trait ModuleA(Movable, Defaultable, Deinitable):
    comptime ARITY: Int
    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime ACT_DT: DType  # the activation-flow dtype (fp32 = NoAMP)

    @staticmethod
    def make[target: StaticString, INIT: Initializer](
        ctx: Optional[DeviceContext] = None
    ) raises -> Self: ...

    def forward[target: StaticString, B: Int, o: MutOrigin](
        mut self,
        inputs: TensorRefsA[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises: ...


# ── cast kernel (fp32 -> ACT_DT) for the bias and the entry boundary ─────
def _cast_kernel[SRC: DType, DST: DType, N: Int](
    src: Tile[SRC, N],
    dst: Tile[DST, N],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i].cast[DST]()


def _bias_add_kernel[ADT: DType, B: Int, OUT: Int](
    acc: Tile[ADT, B, OUT],
    bias: Tile[ADT, OUT],
):
    var idx = Int(global_idx.x)
    if idx < B * OUT:
        acc[idx // OUT, idx % OUT] += bias[idx % OUT]


# ── LinearA: bf16-FLOW linear (input arrives ACT_DT, output ACT_DT) ──────
struct LinearA[IN_: Int, OUT_: Int, ADT: DType = DT](ModuleA):
    comptime ARITY = 1
    comptime IN_DIM = Self.IN_
    comptime OUT_DIM = Self.OUT_
    comptime ACT_DT = Self.ADT
    comptime W_SIZE = Self.IN_ * Self.OUT_

    var weight: Param["weight", True, Self.W_SIZE]  # fp32 master
    var bias: Param["bias", False, Self.OUT_]
    var w_bf: TensorImpl[Self.ADT]   # weight cast to the flow dtype (cached)
    var b_a: TensorImpl[Self.ADT]    # bias cast to the flow dtype
    var _wv: Int

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.OUT_]()
        self.w_bf = TensorImpl[Self.ADT]()
        self.b_a = TensorImpl[Self.ADT]()
        self._wv = -1

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var l = Self()
        l.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        l.bias = Param["bias", False, Self.OUT_].make[target](ctx)
        INIT.init_weight[target](l.weight.val, Self.W_SIZE, Self.IN_, Self.OUT_, ctx)
        INIT.init_bias[target](l.bias.val, Self.OUT_, ctx)
        return l^

    def forward[target: StaticString, B: Int, o: MutOrigin](
        mut self,
        inputs: TensorRefsA[1, o, Self.ADT],
        mut out: TensorImpl[Self.ADT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        var c = ctx.value()
        out.ensure_gpu(c, B * Self.OUT_)
        comptime if Self.ADT == DT:
            # NoAMP: identical to production Linear (fp32 GEMM + bias).
            var xv = TileTensor(x.dev.value(), row_major[B, Self.IN_]())
            var wv = TileTensor(self.weight.val.dev.value(), row_major[Self.IN_, Self.OUT_]())
            var ov = TileTensor(out.dev.value(), row_major[B, Self.OUT_]())
            max_matmul[target="gpu"](ov, xv, wv, c)
            var ol = out.tt["gpu", B, Self.OUT_]()
            var bl = self.bias.val.tt["gpu", Self.OUT_]()
            c.enqueue_function[_bias_add_kernel[Self.ADT, B, Self.OUT_]](
                ol, bl, grid_dim=(B * Self.OUT_ + 255) // 256, block_dim=256)
        else:
            # bf16-FLOW: input ALREADY ADT (no input cast — the whole point);
            # weight cast cached; bias cast to ADT; GEMM ADT->ADT.
            self.w_bf.ensure_gpu(c, Self.W_SIZE)
            self.b_a.ensure_gpu(c, Self.OUT_)
            if self.weight.val.version != self._wv:
                c.enqueue_function[_cast_kernel[DT, Self.ADT, Self.W_SIZE]](
                    self.weight.val.tt["gpu", Self.W_SIZE](),
                    self.w_bf.tt["gpu", Self.W_SIZE](),
                    grid_dim=(Self.W_SIZE + 255) // 256, block_dim=256)
                c.enqueue_function[_cast_kernel[DT, Self.ADT, Self.OUT_]](
                    self.bias.val.tt["gpu", Self.OUT_](),
                    self.b_a.tt["gpu", Self.OUT_](),
                    grid_dim=(Self.OUT_ + 255) // 256, block_dim=256)
                self._wv = self.weight.val.version
            var xv = TileTensor(x.dev.value(), row_major[B, Self.IN_]())
            var wv = TileTensor(self.w_bf.dev.value(), row_major[Self.IN_, Self.OUT_]())
            var ov = TileTensor(out.dev.value(), row_major[B, Self.OUT_]())
            max_matmul[target="gpu"](ov, xv, wv, c)
            var ol = out.tt["gpu", B, Self.OUT_]()
            var bl = self.b_a.tt["gpu", Self.OUT_]()
            c.enqueue_function[_bias_add_kernel[Self.ADT, B, Self.OUT_]](
                ol, bl, grid_dim=(B * Self.OUT_ + 255) // 256, block_dim=256)


# ── Seq2A: 2-stage combinator — intermediate is a TensorImpl[A.ACT_DT] FIELD ─
struct Seq2A[A: ModuleA, B: ModuleA](ModuleA):
    comptime ARITY = 1
    comptime IN_DIM = Self.A.IN_DIM
    comptime OUT_DIM = Self.B.OUT_DIM
    # ACT_DT == B.ACT_DT (== A.ACT_DT by the __init__ assert). Choosing B's means
    # `out` (TensorImpl[Self.ACT_DT]) already matches b.forward's out — no
    # mut-ref rebind. Only the (value-typed) INPUT packs need a rebind to bridge
    # the compiler's refusal to unify A.ACT_DT and B.ACT_DT.
    comptime ACT_DT = Self.B.ACT_DT

    var a: Self.A
    var b: Self.B
    var act0: TensorImpl[Self.A.ACT_DT]   # the KEY: persistent intermediate FIELD

    def __init__(out self):
        comptime assert Self.A.ACT_DT == Self.B.ACT_DT, "Seq2A: child ACT_DT mismatch"
        comptime assert Self.A.OUT_DIM == Self.B.IN_DIM, "Seq2A: dim mismatch"
        self.a = Self.A()
        self.b = Self.B()
        self.act0 = TensorImpl[Self.A.ACT_DT]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.a = Self.A.make[target, INIT](ctx)
        s.b = Self.B.make[target, INIT](ctx)
        return s^

    def forward[target: StaticString, B_: Int, o: MutOrigin](
        mut self,
        inputs: TensorRefsA[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # A: rebind the input pack from Self.ACT_DT(=B) to A.ACT_DT; out=act0 (A).
        var ain = TensorRefsA[1, o, Self.ACT_DT](inputs[0])
        self.a.forward[target, B_](
            rebind[TensorRefsA[Self.A.ARITY, o, Self.A.ACT_DT]](ain),
            self.act0,
            ctx,
        )
        # B: rebind the (act0) pack from A.ACT_DT to B.ACT_DT; out matches (B).
        var bin = TensorRefsA[1, origin_of(self.act0), Self.A.ACT_DT](self.act0)
        self.b.forward[target, B_](
            rebind[TensorRefsA[Self.B.ARITY, origin_of(self.act0), Self.B.ACT_DT]](bin),
            out,
            ctx,
        )


comptime IN = 64
comptime HID = 128
comptime OUT = 32
comptime BATCH = 8


def main() raises:
    print("=" * 70)
    print("ACT_DT threading spike (bf16-flow plumbing)")
    print("=" * 70)
    var c = DeviceContext()

    # ---- (1)+(2) NoAMP fp32: Seq2A vs production Sequential[Linear,Linear] ----
    var spike = Seq2A[LinearA[IN, HID], LinearA[HID, OUT]].make["gpu", Deterministic](Optional(c))
    var prod = Sequential[Linear[IN, HID], Linear[HID, OUT]].make["gpu", Deterministic](Optional(c))

    var x = TensorImpl[DT].alloc(BATCH * IN)
    for i in range(BATCH * IN):
        x.data[i] = Scalar[DT](0.1 + 0.03 * Float64(i % 11))
    x.upload(c)
    var xp = Tensor.alloc(BATCH * IN)
    for i in range(BATCH * IN):
        xp.data[i] = x.data[i]
    xp.upload(c)

    var ys = TensorImpl[DT]()
    var yp = Tensor()
    spike.forward["gpu", BATCH](TensorRefsA[1, origin_of(x), DT](x), ys, Optional(c))
    prod.forward["gpu", BATCH](TensorRefs[1](xp), yp, Optional(c))
    ys.download(c); yp.download(c)
    var md: Scalar[DT] = 0
    for i in range(BATCH * OUT):
        var d = abs(ys.data[i] - yp.data[i])
        if d > md: md = d
    var ok1 = md == Scalar[DT](0)
    print("  (1)(2) NoAMP fp32 vs production Sequential: max|Δ| =", md,
          "BIT-IDENTICAL OK" if ok1 else "FAIL")

    # ---- (3) bf16-flow RUNS (intermediate act0 is bf16) ----
    var sbf = Seq2A[LinearA[IN, HID, BF16], LinearA[HID, OUT, BF16]].make["gpu", Deterministic](Optional(c))
    var xb = TensorImpl[BF16].alloc(BATCH * IN)
    for i in range(BATCH * IN):
        xb.data[i] = x.data[i].cast[BF16]()
    xb.upload(c)
    var yb = TensorImpl[BF16]()
    sbf.forward["gpu", BATCH](TensorRefsA[1, origin_of(xb), BF16](xb), yb, Optional(c))
    yb.download(c)
    c.synchronize()
    # bf16-flow numeric parity vs the fp32 production output (yp). On NVIDIA this
    # should be a few % (real bf16-flow correctness); on Apple it's garbage from
    # the Metal bf16 linalg bug. Printed, not hard-asserted, so the Apple dev run
    # still passes on the (dtype-agnostic) NoAMP-identical gate below.
    var mdb: Scalar[DT] = 0
    var mrb: Scalar[DT] = 0
    for i in range(BATCH * OUT):
        var d = abs(yb.data[i].cast[DT]() - yp.data[i])
        if d > mdb: mdb = d
        if abs(yp.data[i]) > mrb: mrb = abs(yp.data[i])
    if mrb < Scalar[DT](1e-6): mrb = Scalar[DT](1e-6)
    var relb = mdb / mrb
    print("  (3) bf16-flow vs fp32: rel.err =", relb,
          "OK" if relb < Scalar[DT](0.05) else "FAIL (expected on Apple; want OK on NVIDIA)")

    assert_true(ok1, "ACT_DT spike: NoAMP bit-identical")
    print("ALL PASSED (NoAMP-identical gate; see (3) for the NVIDIA bf16 check)")
