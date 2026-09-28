# +--------------------------------------------------------------------------+ #
# | Reference-dump plumbing for the ACT gates
# +--------------------------------------------------------------------------+ #
"""Read a `tools/act/dump_act_reference.py` dump, and push its weights into a
module BY NAME.

Gating a port against a reference means running both on the SAME weights.
Positional loading would do it in ten fewer lines and would also silently
succeed when two same-sized params swap — the q/k/v projections in a DETR layer
are all `[DIM, DIM]`, so exactly the confusion that matters most is the one a
positional loader cannot see. Everything here is addressed by the dotted
`for_each_param` name, and a name present on one side but not the other is an
error naming the name.

Originally test-only. It is now also the path by which ImageNet-pretrained
ResNet18 weights reach the backbone (`LoadPrefixedParams` +
`ACTTrainer.load_backbone`), so it ships and is on a training path. The gate
usage is unchanged.

## Dump format

    <dir>/manifest.txt      `name<TAB>d0,d1,...` per array
    <dir>/<name>.bin        raw little-endian float32, C order

## Weight layout

⚠ **torch `nn.Linear.weight` is `[out_features, in_features]`.** The dump
script transposes to whatever this framework's `Linear` expects and records the
transposed shape, so `RefDump` itself does no reinterpretation — a blob is
written in the destination's layout or the load fails on size. The one place
that convention is decided is `dump_act_reference.py`.
"""

from noeira.core.bytes import string_from_bytes

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor, ParamVisitorRT


struct RefDump(Movable & Deinitable):
    """A reference dump directory: names, sizes, and lazily-read blobs."""

    var root: String
    var names: List[String]
    var sizes: List[Int]

    def __init__(out self, var root: String) raises:
        self.root = root^
        self.names = List[String]()
        self.sizes = List[Int]()
        # ⚠ BYTES — see `core/bytes.mojo`. A param NAME is ASCII in every
        # dump this reader has seen, but it is written by a Python tool from
        # whatever `for_each_param` yields, so this does not get to assume it.
        var text: String
        with open(String(self.root) + "/manifest.txt", "r") as f:
            var raw = f.read_bytes()
            var tb = List[UInt8]()
            for i in range(len(raw)):
                tb.append(raw[i])
            text = string_from_bytes(tb)
            _ = raw^
        var lb = List[UInt8]()
        var bytes = text.as_bytes()
        for i in range(len(bytes)):
            if bytes[i] == UInt8(ord("\n")):
                if len(lb) > 0:
                    self._add(string_from_bytes(lb))
                lb = List[UInt8]()
            else:
                lb.append(bytes[i])
        if len(lb) > 0:
            self._add(string_from_bytes(lb))

    def __init__(out self, *, deinit move: Self):
        self.root = move.root^
        self.names = move.names^
        self.sizes = move.sizes^

    def _add(mut self, line: String) raises:
        """`name<TAB>d0,d1,...` -> (name, product of dims)."""
        var b = line.as_bytes()
        var tab = -1
        for i in range(len(b)):
            if b[i] == UInt8(ord("\t")):
                tab = i
                break
        if tab < 0:
            raise Error("RefDump: manifest line has no tab: " + line)
        var nb = List[UInt8]()
        for i in range(tab):
            nb.append(b[i])
        var n = 1
        # ⚠ THE DIMENSIONS ARE DIGITS, so `cur` could stay a String — it is a
        # List here only so the two loops read the same way. `Int(...)` of a
        # byte-built string is identical for ASCII digits.
        var cb = List[UInt8]()
        for i in range(tab + 1, len(b)):
            if b[i] == UInt8(ord(",")):
                n *= Int(string_from_bytes(cb))
                cb = List[UInt8]()
            else:
                cb.append(b[i])
        if len(cb) > 0:
            n *= Int(string_from_bytes(cb))
        self.names.append(string_from_bytes(nb))
        self.sizes.append(n)

    def has(self, name: String) -> Bool:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return True
        return False

    def size_of(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.sizes[i]
        raise Error("RefDump: no array named '" + name + "'")

    def get(self, name: String) raises -> List[Scalar[DT]]:
        var n = self.size_of(name)
        var out = List[Scalar[DT]](unsafe_uninit_length=n)
        with open(String(self.root) + "/" + name + ".bin", "r") as f:
            var bytes = f.read_bytes()
            if len(bytes) != n * 4:
                raise Error(
                    "RefDump: " + name + ".bin is " + String(len(bytes))
                    + " bytes but the manifest says " + String(n * 4)
                )
            var p = bytes.unsafe_ptr().unsafe_bitcast[Scalar[DT]]()
            for i in range(n):
                out[i] = p[unsafe_offset=i]
            _ = bytes^
        return out^


comptime ACT_CONTENT_QUERY_PARAM: StaticString = "tgt0.queries"
"""The decoder's content queries: learned here, zeros in the reference."""


struct LoadRefParams[PREFIX: StaticString](ParamVisitor, ParamVisitorRT):
    """`ParamVisitor` filling each param from `<PREFIX><dotted name>`.

    Records what it touched so the caller can assert full coverage. A param the
    dump does not mention keeps its random init, and the comparison — which is
    supposed to be running identical weights on both sides — then reads as a
    small numerical disagreement rather than as the missing weight it is.
    """

    var dump: RefDump
    var loaded: List[String]
    var missing: List[String]

    def __init__(out self, var dump: RefDump):
        self.dump = dump^
        self.loaded = List[String]()
        self.missing = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.dump = move.dump^
        self.loaded = move.loaded^
        self.missing = move.missing^

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var key = String(Self.PREFIX) + name
        if not self.dump.has(key):
            # ⚠ THE ONE PARAM WITH NO REFERENCE COUNTERPART, FILLED WITH THE
            # REFERENCE'S OWN VALUE. The decoder's target tokens are learned
            # content queries here (`loss_graph.mojo`, `tgt0`, 2026-09-21: the
            # reference's zeros made every fit predict one action for all 40
            # chunk positions). The reference's `tgt = torch.zeros_like(
            # query_embed)` (`transformer.py:72`) IS a content embedding of
            # zeros, so filling this param with zeros runs the reference's
            # decoder exactly, and every parity gate keeps its meaning. Any
            # OTHER absent name is still reported as missing.
            if name == ACT_CONTENT_QUERY_PARAM:
                var zeros = List[Scalar[DT]](length=n, fill=Scalar[DT](0))
                _fill(param, zeros, ctx)
                self.loaded.append(key^)
                return
            self.missing.append(key^)
            return
        var vals = self.dump.get(key)
        if len(vals) != n:
            raise Error(
                "LoadRefParams: '" + key + "' has " + String(len(vals))
                + " values but the param holds " + String(n)
            )
        _fill(param, vals, ctx)
        self.loaded.append(key^)

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
struct LoadPrefixedParams[GRAPH_PREFIX: StaticString, DUMP_PREFIX: StaticString](
    ParamVisitor
):
    """Fill ONE SUBTREE of a graph from a dump that names it differently.

    `LoadRefParams` assumes the dump uses the walked name verbatim, which holds
    when the dump was written for that exact module. It does not hold for
    pretrained ResNet18 weights: the dump names them `rn18in.0.0.0.weight`
    (backbone-local, the same mapping the standalone gate uses) while the ACT
    graph walks them as `vae.feat.0.0.0.0.weight` — `Tokenwise[N_CAM, ...]`
    contributes a naming level of its own.

    So: visit only names under `GRAPH_PREFIX`, and look each one up as
    `DUMP_PREFIX + name[len(GRAPH_PREFIX):]`. Params outside the subtree are
    SKIPPED, not recorded as missing — that is the point of loading a subtree,
    and counting the rest of the model as missing would bury a real miss under
    two hundred expected ones.

    ⚠ Used for `for_each_state` as well as `for_each_param`. BatchNorm running
    statistics are state, not parameters, and pretrained weights carrying the
    framework's init statistics (mean 0, var 1) are not the pretrained network
    — they are a different function that happens to share its convolutions.
    """

    var dump: RefDump
    var loaded: List[String]
    var missing: List[String]
    var skipped: Int

    def __init__(out self, var dump: RefDump):
        self.dump = dump^
        self.loaded = List[String]()
        self.missing = List[String]()
        self.skipped = 0

    def __init__(out self, *, deinit move: Self):
        self.dump = move.dump^
        self.loaded = move.loaded^
        self.missing = move.missing^
        self.skipped = move.skipped

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime GP = String(Self.GRAPH_PREFIX)
        if not name.startswith(GP):
            self.skipped += 1
            return
        var key = String(Self.DUMP_PREFIX) + String(
            name[byte=GP.byte_length() :]
        )
        if not self.dump.has(key):
            self.missing.append(key^)
            return
        var vals = self.dump.get(key)
        if len(vals) != n:
            raise Error(
                "LoadPrefixedParams: '" + key + "' has " + String(len(vals))
                + " values but the param holds " + String(n)
            )
        _fill(param, vals, ctx)
        self.loaded.append(key^)

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
def _fill(
    mut param: Tensor,
    ref vals: List[Scalar[DT]],
    ctx: Optional[DeviceContext],
) raises:
    """Host fill, then upload when the module lives on a device.

    ⚠ The upload is not optional and its absence is silent: the host tensor
    would hold the pretrained weights, every device kernel would keep reading
    the random init, and the run would look like pretraining simply did not
    help.
    """
    param.ensure(len(vals))
    for i in range(len(vals)):
        param.data[i] = vals[i]
    if ctx:
        param.upload_resident(ctx.value())


struct ListParams(ParamVisitor, ParamVisitorRT):
    """Collects `(name, size)` for every param — used to author the dump's name
    mapping and to check it stays in step."""

    var names: List[String]
    var sizes: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.sizes = List[Int]()

    def __init__(out self, *, deinit move: Self):
        self.names = move.names^
        self.sizes = move.sizes^

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.names.append(String(name))
        self.sizes.append(n)

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
    def show(self) raises:
        for i in range(len(self.names)):
            print("  " + self.names[i] + "\t" + String(self.sizes[i]))
