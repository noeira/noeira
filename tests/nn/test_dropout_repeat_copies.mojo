"""Dropout inside `Repeat`: every copy draws its own masks.

`Repeat[N, Block]` holds N values of ONE type, so a `Dropout` in the block
has one comptime SEED in all N copies, and with the same counter the copies
drew the SAME mask at every forward. In the GPT (`GPTDrop`, `GPTDropTied`,
`GPTMarked`) every block's attention-out dropout dropped the same entries as
the other blocks', and so did the FFN-out dropouts. `Repeat.make` now sends
copy i `set_attr["rng_copy"](i)`, and `Dropout` draws copy i's masks from
Philox subsequence i. Copy 0, and every dropout outside a `Repeat`, stays on
subsequence 0: the stream it drew before.

Gates, each over two training forwards (the counter moves every copy by the
same amount, so a second forward must not line the copies up again):
  - `Repeat[3, Dropout]` on the CPU, the GPU and the GPU's bf16 flow: the
    copies' masks differ pairwise and copy 0's is a lone `Dropout`'s; the
    three targets draw the same masks;
  - `Repeat[2, Repeat[2, Dropout]]`: four different masks (a nested stack
    does not fold its copies back together);
  - the GPT (`GPTDropTied`, 3 blocks; on the GPU, else the CPU): the blocks'
    attention-out masks differ pairwise, and so do their FFN-out masks;
    block 0's two and the input dropout's are lone `Dropout`s';
  - every mask keeps 1 - p of its entries, within 5 sigma.

Run: pixi run -e apple mojo run -I . tests/nn/test_dropout_repeat_copies.mojo
"""

from std.math import sqrt
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.dropout import Dropout
from noeira.nn.combinators.repeat import Repeat
from noeira.nn.models.gpt import GPTDropTied, gpt_wire_tie


comptime BF16 = DType.bfloat16
comptime P = 0.5
comptime SEED = UInt64(7)
comptime DIM = 256
comptime B = 4
comptime N = B * DIM
comptime CALLS = 2

# A GPT with three blocks: GPTDropTied's three dropout points.
comptime VOCAB = 8
comptime SEQ = 8
comptime EMB = 16
comptime HEADS = 2
comptime LAYERS = 3
comptime FFM = 2
comptime SEED_BASE = UInt64(0xC0FFEE)
comptime GPT = GPTDropTied[
    VOCAB, SEQ, EMB, HEADS, LAYERS, FFM, True, P, SEED_BASE, True
]
comptime W = SEQ * EMB
"""The width of every GPT dropout (input, attention-out, FFN-out)."""


struct _Gate(Movable):
    var failed: List[String]

    def __init__(out self):
        self.failed = List[String]()

    def check(mut self, ok: Bool, what: String):
        print("    " + ("ok   " if ok else "FAIL ") + what)
        if not ok:
            self.failed.append(what)


def _keep[dt: DType](
    mut m: TensorImpl[dt], ctx: Optional[DeviceContext]
) raises -> List[Bool]:
    """Which entries a cached mask keeps (a device mask is downloaded)."""
    if ctx:
        m.download(ctx.value())
    var keep = List[Bool](capacity=m.n)
    for i in range(m.n):
        keep.append(m.data[i] != Scalar[dt](0))
    return keep^


def _same(a: List[Bool], b: List[Bool]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _distinct(masks: List[List[Bool]]) -> Bool:
    """Every pair of masks differs."""
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if _same(masks[i], masks[j]):
                return False
    return True


def _rate_ok(masks: List[List[Bool]]) -> Bool:
    """Every mask keeps 1 - P of its entries, within 5 sigma."""
    for k in range(len(masks)):
        var kept = 0
        for i in range(len(masks[k])):
            if masks[k][i]:
                kept += 1
        var n = Float64(len(masks[k]))
        var off = abs(Float64(kept) / n - (1.0 - P))
        if off > 5.0 * sqrt(P * (1.0 - P) / n):
            return False
    return True


def _hash(k: List[Bool]) -> UInt64:
    """FNV-1a over a keep pattern: a mask's fingerprint, comparable across
    targets and across builds."""
    var h = UInt64(0xCBF29CE484222325)
    for i in range(len(k)):
        h = (h ^ UInt64(1 if k[i] else 0)) * UInt64(0x100000001B3)
    return h


def _same_prints(a: List[UInt64], b: List[UInt64]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _ones[dt: DType](
    n: Int, ctx: Optional[DeviceContext]
) raises -> TensorImpl[dt]:
    var x = TensorImpl[dt].alloc(n)
    for i in range(n):
        x.data[i] = Scalar[dt](1)
    if ctx:
        x.upload(ctx.value())
    return x^


def _repeat3[target: StaticString, ADT: DType](
    mut g: _Gate, ctx: Optional[DeviceContext]
) raises -> List[UInt64]:
    """`Repeat[3, Dropout]` against a lone `Dropout` of the same type. Returns
    the fingerprints, forward by forward: the lone mask, then copies 0, 1, 2.
    """
    comptime D = Dropout[DIM, P, SEED, ADT]
    var stack = Repeat[3, D].make[target, Deterministic](ctx)
    var lone = D.make[target, Deterministic](ctx)
    var x = _ones[ADT](N, ctx)
    var y = TensorImpl[ADT]()
    var yl = TensorImpl[ADT]()
    var prints = List[UInt64]()
    for call in range(CALLS):
        stack.forward[target, B](TensorRefs[1, ADT=ADT](x), y, ctx)
        lone.forward[target, B](TensorRefs[1, ADT=ADT](x), yl, ctx)
        var kl = _keep(lone.cache_mask, ctx)
        var copies = List[List[Bool]]()
        for i in range(3):
            copies.append(_keep(stack.children[i].cache_mask, ctx))
        var at = " (forward " + String(call) + ")"
        g.check(_same(copies[0], kl), "copy 0 draws the lone Dropout's mask" + at)
        g.check(_distinct(copies), "the 3 copies draw 3 different masks" + at)
        g.check(_rate_ok(copies), "each copy keeps 1 - p of its entries" + at)
        prints.append(_hash(kl))
        for i in range(3):
            prints.append(_hash(copies[i]))
        print(
            "      fingerprints" + at + ": lone", prints[4 * call],
            " copies", prints[4 * call + 1], prints[4 * call + 2],
            prints[4 * call + 3],
        )
    return prints^


def _nested(mut g: _Gate) raises:
    """`Repeat[2, Repeat[2, Dropout]]` on the CPU: four copies, four masks."""
    comptime D = Dropout[DIM, P, SEED]
    var stack = Repeat[2, Repeat[2, D]].make["cpu", Deterministic](None)
    var lone = D.make["cpu", Deterministic](None)
    var x = _ones[DT](N, None)
    var y = Tensor()
    var yl = Tensor()
    for call in range(CALLS):
        stack.forward["cpu", B](TensorRefs[1](x), y, None)
        lone.forward["cpu", B](TensorRefs[1](x), yl, None)
        var copies = List[List[Bool]]()
        for i in range(2):
            for j in range(2):
                copies.append(
                    _keep(stack.children[i].children[j].cache_mask, None)
                )
        var at = " (forward " + String(call) + ")"
        g.check(
            _same(copies[0], _keep(lone.cache_mask, None)),
            "copy (0, 0) draws the lone Dropout's mask" + at,
        )
        g.check(_distinct(copies), "the 4 nested copies draw 4 different masks" + at)


def _gpt[target: StaticString](
    mut g: _Gate, ctx: Optional[DeviceContext]
) raises:
    """The GPT's three blocks against lone `Dropout`s of the GPT's types."""
    var net = GPT.make[target, Deterministic](ctx)
    gpt_wire_tie[
        target, VOCAB, SEQ, EMB, HEADS, LAYERS, FFM, True, P, SEED_BASE, True
    ](net)
    var x = Tensor.alloc(B * SEQ * VOCAB)
    for i in range(B * SEQ * VOCAB):
        x.data[i] = Scalar[DT](0)
    for b in range(B):
        for t in range(SEQ):
            x.data[(b * SEQ + t) * VOCAB + (b + 3 * t) % VOCAB] = Scalar[DT](1)
    if ctx:
        x.upload(ctx.value())
    var out = Tensor()
    var lone_in = Dropout[W, P, SEED_BASE].make[target, Deterministic](ctx)
    var lone_attn = Dropout[W, P, SEED_BASE + 1].make[target, Deterministic](
        ctx
    )
    var lone_ffn = Dropout[W, P, SEED_BASE + 2].make[target, Deterministic](
        ctx
    )
    var ones = _ones[DT](B * W, ctx)
    var y = Tensor()
    for call in range(CALLS):
        net.forward[target, B](TensorRefs[1](x), out, ctx)
        lone_in.forward[target, B](TensorRefs[1](ones), y, ctx)
        lone_attn.forward[target, B](TensorRefs[1](ones), y, ctx)
        lone_ffn.forward[target, B](TensorRefs[1](ones), y, ctx)
        var attn = List[List[Bool]]()
        var ffn = List[List[Bool]]()
        for L in range(LAYERS):
            # Block L: [0] Residual(LN, MHADrop), the Dropout is MHADrop's
            # child [3]; [1] Residual(LN, FFNDrop), FFNDrop's child [1].
            attn.append(
                _keep(
                    net.children[3].children[L].children[0]
                    .inner.children[1].children[3].cache_mask,
                    ctx,
                )
            )
            ffn.append(
                _keep(
                    net.children[3].children[L].children[1]
                    .inner.children[1].children[1].cache_mask,
                    ctx,
                )
            )
        var k_in = _keep(net.children[2].cache_mask, ctx)
        var at = " (forward " + String(call) + ")"
        g.check(_distinct(attn), "the 3 blocks' attention-out masks differ" + at)
        g.check(_distinct(ffn), "the 3 blocks' FFN-out masks differ" + at)
        g.check(
            _same(attn[0], _keep(lone_attn.cache_mask, ctx))
            and _same(ffn[0], _keep(lone_ffn.cache_mask, ctx)),
            "block 0 draws lone Dropouts' masks" + at,
        )
        g.check(
            _same(k_in, _keep(lone_in.cache_mask, ctx)),
            "the input dropout draws a lone Dropout's mask" + at,
        )
        g.check(
            _rate_ok(attn) and _rate_ok(ffn),
            "every block mask keeps 1 - p of its entries" + at,
        )
        print(
            "      fingerprints" + at + ": input", _hash(k_in),
            " attention-out", _hash(attn[0]), _hash(attn[1]), _hash(attn[2]),
            " FFN-out", _hash(ffn[0]), _hash(ffn[1]), _hash(ffn[2]),
        )


def main() raises:
    print("=" * 70)
    print("Dropout inside Repeat: every copy draws its own masks")
    print("=" * 70)
    var g = _Gate()
    print("  Repeat[3, Dropout], CPU")
    var cpu = _repeat3["cpu", DT](g, None)
    print("  Repeat[2, Repeat[2, Dropout]], CPU")
    _nested(g)
    comptime if has_accelerator():
        var ctx = DeviceContext()
        print("  Repeat[3, Dropout], GPU")
        var gpu = _repeat3["gpu", DT](g, Optional(ctx))
        print("  Repeat[3, Dropout], GPU, bf16 flow")
        var bf16 = _repeat3["gpu", BF16](g, Optional(ctx))
        g.check(_same_prints(gpu, cpu), "the GPU draws the CPU's masks")
        g.check(_same_prints(bf16, gpu), "the bf16 flow draws the fp32 masks")
        print("  GPTDropTied, 3 blocks, GPU")
        _gpt["gpu"](g, Optional(ctx))
    else:
        print("  GPTDropTied, 3 blocks, CPU (no accelerator)")
        _gpt["cpu"](g, None)
    if len(g.failed) > 0:
        print(String(len(g.failed)) + " check(s) FAILED")
    assert_true(len(g.failed) == 0, "Dropout copies inside Repeat")
    print("ALL PASSED")
