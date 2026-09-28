"""Checkpoint — named-section param/state + optimizer-moment save/load.

Walks a Module's Params via `for_each_param` then States via `for_each_state`,
writing one named section per field. Two formats:

v3 (CURRENT — what every save now writes):

    storage-ckpt v3
    P <dotted-name> <size> <has_moments:0|1>\n
    <size raw Scalar[DT] payload bytes>[<m bytes><v bytes>]
    S <dotted-name> <size>\n
    <size raw Scalar[DT] payload bytes>
    ...
    K <name>\n                      (zero or more, AFTER every P/S section)
    <8 raw Float64 bytes>

`K` sections carry the training state that is not a tensor: ε, α, step
counters, Adam's `t` and bias-correction powers (`CheckpointScalars`). Before
they existed, v3 had nowhere to put a number, so four trainers stayed on v2 to
append `key=value` lines and every v3 trainer lost that state on resume. The
value is raw Float64 bytes, not text: `Float64(String)` reads one ULP low, and
a step counter stored through `Scalar[DT]` (float32) stops counting at 2^24.
A file with no `K` section is still a valid v3 file — its scalars are empty
and every `get` returns the caller's default.

v2 (LEGACY — still readable; loaders dispatch on the header line): identical
section headers but one ASCII float per line. v2 hit its ceiling at DreamerV3
size200m: ~5 GB of text per save, silently TRUNCATED at the single-write(2)
syscall cap (0x7FFFF000 ≈ 2 GiB) → corrupt checkpoints, and the per-line
`List[String]` loader could not have held it anyway. v3 payloads are raw
little-endian bytes (3× smaller, no atof/String churn) and ALL file I/O goes
through explicit ≤1 GiB chunks (`_write_file_bytes`/`_read_file_bytes`).

The dotted name (from the A2 name-threading walker, e.g. "0.weight") is VALIDATED
against the in-memory walk order on load — a name/size mismatch raises, catching
topology drift between save and load (the legacy named-section guarantee the
positional v1 format lacked).

Optimizer moments (Adam's per-param `m`/`v`, co-located on the Param) ride the
same param section when populated (`m.n >= N`), enabling exact training resume.
`save_moments=False` writes a model-only checkpoint (moments skipped). GPU params
download on save / upload on load.
"""

from noeira.core.bytes import string_from_bytes

from max.gpu.host import DeviceContext
from std.memory import bitcast, unsafe_memcpy
from std.sys.info import size_of

from noeira.io.fileio import read_file_bytes, write_file_atomic
from noeira.nn.constants import DT
from .tensor import Tensor
from .param import ParamVisitor, ParamVisitorRT, walk_params, ParamVisitorRef
from .param import ParamWalkable

def _write_file_bytes(var path: String, content: List[UInt8]) raises:
    """Chunked, ATOMIC file write — see `noeira/io/fileio.mojo` for both
    rules and why they are not optional (a single `write(2)` silently stops at
    ~2 GiB, which is how v2 produced truncated checkpoints; `rename(2)` is what
    keeps a crash mid-save from destroying the previous good one).

    Kept as a forward rather than inlined again: `io/safetensors.mojo` needs
    the same two rules, and a rule written inline twice is this repo's most
    frequent defect shape."""
    write_file_atomic(path^, content)


def _read_file_bytes(path: String) raises -> List[UInt8]:
    """Chunked file read — `read(2)` has the same single-call cap as write."""
    return read_file_bytes(path)


def _bytes_append_str(mut buf: List[UInt8], s: String):
    var sb = s.as_bytes()
    var old = len(buf)
    buf.resize(old + len(sb), 0)
    unsafe_memcpy(dest=buf.unsafe_ptr().unsafe_offset(old), src=sb.unsafe_ptr(), count=len(sb))


def _bytes_append_vals(mut buf: List[UInt8], t: Tensor, n: Int):
    comptime SB = size_of[Scalar[DT]]()
    var old = len(buf)
    buf.resize(old + n * SB, 0)
    unsafe_memcpy(
        dest=buf.unsafe_ptr().unsafe_offset(old),
        src=t.data.unsafe_ptr().unsafe_bitcast[UInt8](),
        count=n * SB,
    )


def _bytes_append_f64(mut buf: List[UInt8], x: Float64):
    var old = len(buf)
    buf.resize(old + 8, 0)
    var bits = bitcast[DType.uint64](x)
    for i in range(8):
        buf[old + i] = UInt8((bits >> UInt64(8 * i)) & 0xFF)


struct CheckpointScalars(Copyable, Defaultable, Movable, Sized):
    """The named non-tensor state of a checkpoint — one `K` section each.

    Order is kept (sections are written in insertion order); `set` on an
    existing name overwrites it rather than writing the name twice.
    """
    var names: List[String]
    var values: List[Float64]

    def __init__(out self):
        self.names = List[String]()
        self.values = List[Float64]()

    def __len__(self) -> Int:
        return len(self.names)

    def _find(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def set(mut self, name: String, value: Float64) raises:
        # A space or newline would split the `K <name>` header on load.
        if name.byte_length() == 0 or " " in name or "\n" in name:
            raise Error(
                "checkpoint scalar: invalid name `" + name
                + "` (empty, or holds a space / newline)"
            )
        var i = self._find(name)
        if i >= 0:
            self.values[i] = value
            return
        self.names.append(name)
        self.values.append(value)

    def set_int(mut self, name: String, value: Int) raises:
        """Exact for |value| < 2^53 — every counter this tree keeps."""
        self.set(name, Float64(value))

    def has(self, name: String) -> Bool:
        return self._find(name) >= 0

    def get(self, name: String, default: Float64) -> Float64:
        var i = self._find(name)
        return self.values[i] if i >= 0 else default

    def get_int(self, name: String, default: Int) -> Int:
        var i = self._find(name)
        return Int(self.values[i]) if i >= 0 else default


def _is_v3_header(bytes: List[UInt8]) -> Bool:
    var tag = String("storage-ckpt v3")
    var tb = tag.as_bytes()
    if len(bytes) < len(tb):
        return False
    for i in range(len(tb)):
        if bytes[i] != tb[i]:
            return False
    return True


def _split_lines(content: String) -> List[String]:
    var lines = List[String]()
    # ⚠ BYTES — see `core/bytes.mojo`.
    var cur = List[UInt8]()
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(string_from_bytes(cur))
            cur = List[UInt8]()
        else:
            cur.append(c)
    if len(cur) > 0:
        lines.append(string_from_bytes(cur))
    return lines^


struct LegacyV2CheckpointWriter(ParamVisitor, ParamVisitorRT):
    """The v2 TEXT writer — kept ONLY to make v2 fixtures for the tests that
    prove old files still load. Nothing in the library writes v2 any more:
    use `BinaryCheckpointWriter` (+ `write_model`, `write_scalars`) or
    `save_params` / `save_params_multi`.

    ⚠ The name is the guard. Twelve trainers kept writing v2 after v3 landed
    because this struct was still called `CheckpointWriter`, the obvious
    name, and v3 had no place for their `key=value` scalars; the files were
    3× larger, non-atomic, and truncated at 2 GiB.

    Appends a named section per visited Param/State. `mode`: 0 = Param (P,
    with optional moments), 1 = State (S). `save_moments` gates m/v output."""
    var content: String
    var mode: Int
    var save_moments: Bool

    def __init__(out self, save_moments: Bool = True):
        self.content = String("storage-ckpt v2\n")
        self.mode = 0
        self.save_moments = save_moments

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        if self.mode == 1:  # State
            self.content += "S " + name + " " + String(n) + "\n"
            for i in range(n):
                self.content += String(param.data[i]) + "\n"
            return
        # Param: include moments when populated (optimizer has stepped).
        var has_m = self.save_moments and m.n >= n and v.n >= n
        comptime if target == "gpu":
            if has_m:
                m.download(ctx.value())
                v.download(ctx.value())
        self.content += (
            "P " + name + " " + String(n) + " " + ("1" if has_m else "0") + "\n"
        )
        for i in range(n):
            self.content += String(param.data[i]) + "\n"
        if has_m:
            for i in range(n):
                self.content += String(m.data[i]) + "\n"
            for i in range(n):
                self.content += String(v.data[i]) + "\n"

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
struct CheckpointReader(ParamVisitor, ParamVisitorRT):
    """Consumes one named section per visited Param/State, validating the
    section kind + dotted name + size against the in-memory walk (topology-drift
    catch). Restores values and, for Params, the m/v moments if present."""
    var lines: List[String]
    var cur: Int
    var mode: Int

    def __init__(out self, var lines: List[String]):
        self.lines = lines^
        self.cur = 0
        self.mode = 0

    def _next(mut self) raises -> String:
        if self.cur >= len(self.lines):
            raise Error("checkpoint: unexpected end of file")
        var s = self.lines[self.cur]
        self.cur += 1
        return s

    def finish(self) raises:
        """Every line consumed. ⚠ ALSO THE MENTION THAT KEEPS THE READER ALIVE
        across the state pass — see `BinaryCheckpointReader.finish`."""
        if self.cur != len(self.lines):
            raise Error(
                "checkpoint: " + String(len(self.lines) - self.cur)
                + " line(s) left unread — the file holds tensors this build"
                " does not walk (topology drift)"
            )

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var hdr = self._next()
        var toks = hdr.split(" ")
        var expected_kind = String("S") if self.mode == 1 else String("P")
        if len(toks) < 3 or toks[0] != expected_kind:
            raise Error(
                "checkpoint: expected '" + expected_kind + " " + name
                + "' section, got header `" + hdr + "`"
            )
        if toks[1] != name:
            raise Error(
                "checkpoint: name mismatch — model expects `" + name
                + "`, checkpoint has `" + toks[1] + "` (topology drift)"
            )
        if atol(toks[2]) != n:
            raise Error(
                "checkpoint: size mismatch for `" + name + "` — model "
                + String(n) + ", checkpoint " + toks[2]
            )
        for i in range(n):
            param.data[i] = Scalar[DT](atof(self._next()))
        # ⚠ RESTORING A WEIGHT IS A WRITE, so it must advance `version` — the
        # same contract the optimizer honours via `ParamVersionBump`. Leaves
        # cache DERIVED copies of the weight gated on this counter (`w_pad`,
        # the K-alignment pad; `w_bf`, the AMP bf16 recast), and without the
        # bump a `make -> forward -> load_state -> forward` sequence keeps
        # serving the PRE-LOAD weight: the checkpoint loads, reports success,
        # and is silently ignored. A viewer switching checkpoints after it has
        # already acted hits exactly that.
        param.version += 1
        if self.mode == 0 and len(toks) >= 4 and toks[3] == "1":
            m.ensure(n)
            v.ensure(n)
            for i in range(n):
                m.data[i] = Scalar[DT](atof(self._next()))
            for i in range(n):
                v.data[i] = Scalar[DT](atof(self._next()))
            comptime if target == "gpu":
                m.upload_resident(ctx.value())
                v.upload_resident(ctx.value())
        comptime if target == "gpu":
            # ⚠ `upload_resident`, NOT `upload`. `upload` RECREATES the device
            # buffer, and under an adopted `ParamArena` this Param's val/grd/m/v
            # ARE sub-buffers of the optimizer's contiguous arenas — a fresh
            # buffer silently detaches the param from the arena, after which the
            # grouped step updates arena memory the model no longer reads. These
            # are fixed-size slabs, so the resident path is also two fewer
            # synchronizations per parameter.
            param.upload_resident(ctx.value())

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
struct BinaryCheckpointWriter(ParamVisitor, ParamVisitorRT):
    """V3 twin of `LegacyV2CheckpointWriter`: text section headers, raw-byte payloads.
    `mode`: 0 = Param (P, with optional moments), 1 = State (S)."""
    var content: List[UInt8]
    var mode: Int
    var save_moments: Bool

    def __init__(out self, save_moments: Bool = True):
        self.content = List[UInt8]()
        _bytes_append_str(self.content, String("storage-ckpt v3\n"))
        self.mode = 0
        self.save_moments = save_moments

    def write_scalars(mut self, scalars: CheckpointScalars):
        """Append one `K` section per scalar. Call AFTER the last P/S section:
        the reader walks the tensors first and only then reads the scalars."""
        for i in range(len(scalars.names)):
            _bytes_append_str(self.content, "K " + scalars.names[i] + "\n")
            _bytes_append_f64(self.content, scalars.values[i])

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        if self.mode == 1:  # State
            _bytes_append_str(
                self.content, "S " + name + " " + String(n) + "\n"
            )
            _bytes_append_vals(self.content, param, n)
            return
        var has_m = self.save_moments and m.n >= n and v.n >= n
        comptime if target == "gpu":
            if has_m:
                m.download(ctx.value())
                v.download(ctx.value())
        _bytes_append_str(
            self.content,
            "P " + name + " " + String(n) + " "
            + ("1" if has_m else "0") + "\n",
        )
        _bytes_append_vals(self.content, param, n)
        if has_m:
            _bytes_append_vals(self.content, m, n)
            _bytes_append_vals(self.content, v, n)

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
struct BinaryCheckpointReader(ParamVisitor, ParamVisitorRT):
    """V3 twin of `CheckpointReader`: byte-cursor over the whole file, same
    name/size/topology validation as v2."""
    var bytes: List[UInt8]
    var cur: Int
    var mode: Int
    var seen: Int
    """Sections consumed, and `want` the one being read — both only so that
    running out of file NAMES what the model was asking for. "unexpected end
    of file" alone sent a real deployment looking for a truncated download
    that was byte-perfect; the file simply held fewer tensors than the model
    walked."""
    var want: String

    def __init__(out self, var bytes: List[UInt8]):
        self.bytes = bytes^
        self.cur = 0
        self.mode = 0
        self.seen = 0
        self.want = String("")
        # Skip the "storage-ckpt v3" header line.
        while self.cur < len(self.bytes) and self.bytes[self.cur] != UInt8(10):
            self.cur += 1
        if self.cur < len(self.bytes):
            self.cur += 1

    def _eof(self, what: String) -> String:
        return (
            "checkpoint: unexpected end of file " + what + " — the model asked"
            " for `" + self.want + "` after " + String(self.seen)
            + " section(s), and the file has no more. The checkpoint is"
            " COMPLETE and holds fewer tensors than this build walks: it was"
            " written by a different model configuration."
            " [cursor " + String(self.cur) + " of " + String(len(self.bytes))
            + " bytes, pass " + ("state" if self.mode == 1 else "params") + "]"
        )

    def _next_line(mut self) raises -> String:
        if self.cur >= len(self.bytes):
            raise Error(self._eof(String("reading a section header")))
        # ⚠ BYTES — see `core/bytes.mojo`.
        var o = List[UInt8]()
        while self.cur < len(self.bytes) and self.bytes[self.cur] != UInt8(10):
            o.append(self.bytes[self.cur])
            self.cur += 1
        var s = string_from_bytes(o)
        if self.cur < len(self.bytes):
            self.cur += 1  # consume '\n'
        return s^

    def _take_vals(mut self, mut t: Tensor, n: Int) raises:
        comptime SB = size_of[Scalar[DT]]()
        if self.cur + n * SB > len(self.bytes):
            raise Error(self._eof(String("reading ") + String(n) + " values"))
        unsafe_memcpy(
            dest=t.data.unsafe_ptr().unsafe_bitcast[UInt8](),
            src=self.bytes.unsafe_ptr().unsafe_offset(self.cur),
            count=n * SB,
        )
        self.cur += n * SB

    def read_scalars(mut self) raises -> CheckpointScalars:
        """Consume the trailing `K` sections. Call after the last tensor walk
        and before `finish`. A file written before `K` existed has none, and
        returns an empty set."""
        var out = CheckpointScalars()
        while self.cur < len(self.bytes):
            var hdr = self._next_line()
            var toks = hdr.split(" ")
            if len(toks) != 2 or String(toks[0]) != "K":
                raise Error(
                    "checkpoint: section `" + hdr + "` left unread after "
                    + String(self.seen) + " section(s) — only `K` scalar"
                    " sections may follow the last tensor, so the file holds"
                    " tensors this build does not walk (topology drift)"
                )
            if self.cur + 8 > len(self.bytes):
                raise Error(
                    "checkpoint: truncated scalar `" + String(toks[1]) + "`"
                )
            var bits = UInt64(0)
            for i in range(8):
                bits |= UInt64(self.bytes[self.cur + i]) << UInt64(8 * i)
            self.cur += 8
            out.set(String(toks[1]), bitcast[DType.float64](bits))
        return out^

    def finish(self) raises:
        """Every byte accounted for — and the reason this is called at all.

        ⚠⚠ IT ALSO KEEPS THE READER ALIVE. `ParamVisitorRef.of(r)` hands the
        state pass a POINTER to `r`; if `r` is never mentioned afterwards,
        that call is its last use and Mojo destroys it there
        (`_taking_a_view_is_the_owners_last_use`). The state pass then walked
        freed memory: on 2026-09-16 a deployment loaded 253 tensors and failed
        on the first BatchNorm buffer with "unexpected end of file" against a
        file of `0 bytes` — the file was byte-perfect. A mention after the
        pass is what keeps it alive, so this check must not be dropped as
        redundant.

        ⚠ AND IT CATCHES THE OTHER DRIFT: bytes left over mean the file holds
        tensors this build does not walk, which is the same mismatch in the
        other direction and would otherwise pass silently.
        """
        if self.cur != len(self.bytes):
            raise Error(
                "checkpoint: " + String(len(self.bytes) - self.cur)
                + " bytes left unread after " + String(self.seen)
                + " section(s) — the file holds tensors this build does not"
                " walk (topology drift)"
            )

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.want = name
        var hdr = self._next_line()
        self.seen += 1
        var toks = hdr.split(" ")
        var expected_kind = String("S") if self.mode == 1 else String("P")
        if len(toks) < 3 or toks[0] != expected_kind:
            raise Error(
                "checkpoint: expected '" + expected_kind + " " + name
                + "' section, got header `" + hdr + "`"
            )
        if toks[1] != name:
            raise Error(
                "checkpoint: name mismatch — model expects `" + name
                + "`, checkpoint has `" + toks[1] + "` (topology drift)"
            )
        if atol(toks[2]) != n:
            raise Error(
                "checkpoint: size mismatch for `" + name + "` — model "
                + String(n) + ", checkpoint " + toks[2]
            )
        self._take_vals(param, n)
        # See the note in `CheckpointReader.visit` — restoring a weight must
        # advance `version` or the version-gated derived caches (`w_pad`,
        # `w_bf`) keep serving the pre-load weight.
        param.version += 1
        if self.mode == 0 and len(toks) >= 4 and toks[3] == "1":
            m.ensure(n)
            v.ensure(n)
            self._take_vals(m, n)
            self._take_vals(v, n)
            comptime if target == "gpu":
                m.upload_resident(ctx.value())
                v.upload_resident(ctx.value())
        comptime if target == "gpu":
            # ⚠ `upload_resident`, NOT `upload`. `upload` RECREATES the device
            # buffer, and under an adopted `ParamArena` this Param's val/grd/m/v
            # ARE sub-buffers of the optimizer's contiguous arenas — a fresh
            # buffer silently detaches the param from the arena, after which the
            # grouped step updates arena memory the model no longer reads. These
            # are fixed-size slabs, so the resident path is also two fewer
            # synchronizations per parameter.
            param.upload_resident(ctx.value())

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
def write_model[
    target: StaticString, M: ParamWalkable
](
    mut w: BinaryCheckpointWriter,
    mut model: M,
    ctx: Optional[DeviceContext],
    prefix: String = String(""),
) raises:
    """One model's Param then State sections, names under `prefix`. A trainer
    with several nets calls this once per net, then `write_scalars`, then
    `_write_file_bytes` — and loads with `read_model` in the SAME order."""
    w.mode = 0
    walk_params[target](model, w, ctx, prefix)
    w.mode = 1
    var sref = ParamVisitorRef.of[type_of(w), target](w)
    model.for_each_state[target](sref, ctx, prefix)


def read_model[
    target: StaticString, M: ParamWalkable
](
    mut r: BinaryCheckpointReader,
    mut model: M,
    ctx: Optional[DeviceContext],
    prefix: String = String(""),
) raises:
    """Inverse of `write_model`. ⚠ The caller must still mention `r` after its
    last `read_model` — `read_scalars` + `finish` do — or the reader can be
    destroyed under the state pass (see `BinaryCheckpointReader.finish`)."""
    r.mode = 0
    walk_params[target](model, r, ctx, prefix)
    r.mode = 1
    var sref = ParamVisitorRef.of[type_of(r), target](r)
    model.for_each_state[target](sref, ctx, prefix)


def save_params[
    target: StaticString, M: ParamWalkable
](
    mut model: M, path: String,
    ctx: Optional[DeviceContext] = None,
    save_moments: Bool = True,
    scalars: CheckpointScalars = CheckpointScalars(),
) raises:
    """Write a v3 named checkpoint: Params (+ moments if populated), then
    States, then one `K` section per entry of `scalars`."""
    var w = BinaryCheckpointWriter(save_moments)
    w.mode = 0
    walk_params[target](model, w, ctx)
    w.mode = 1
    var _sref1 = ParamVisitorRef.of[type_of(w), target](w)
    model.for_each_state[target](_sref1, ctx)
    w.write_scalars(scalars)
    _write_file_bytes(path, w.content)


def load_params[
    target: StaticString, M: ParamWalkable
](
    mut model: M, path: String, ctx: Optional[DeviceContext] = None
) raises -> CheckpointScalars:
    """Load a named checkpoint (v3 binary, or legacy v2 text — dispatched on
    the header line), validating names/sizes against `model`. Returns the
    file's `K` scalars (empty for v2, and for a v3 file written without any)."""
    var bytes = _read_file_bytes(path)
    if _is_v3_header(bytes):
        var r = BinaryCheckpointReader(bytes^)
        r.mode = 0
        walk_params[target](model, r, ctx)
        r.mode = 1
        var _sref2 = ParamVisitorRef.of[type_of(r), target](r)
        model.for_each_state[target](_sref2, ctx)
        var sc = r.read_scalars()
        r.finish()
        return sc^
    # Legacy v2 text checkpoint.
    var content: String
    with open(path, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    # Drop the format header line.
    var body = List[String]()
    for li in range(len(lines)):
        if lines[li].startswith("storage-ckpt"):
            continue
        body.append(lines[li])
    var r = CheckpointReader(body^)
    r.mode = 0
    walk_params[target](model, r, ctx)
    r.mode = 1
    var _sref3 = ParamVisitorRef.of[type_of(r), target](r)
    model.for_each_state[target](_sref3, ctx)
    r.finish()
    return CheckpointScalars()


def save_params_multi[
    target: StaticString, *Ms: ParamWalkable
](
    path: String,
    ctx: Optional[DeviceContext],
    save_moments: Bool,
    mut *models: *Ms,
    scalars: CheckpointScalars = CheckpointScalars(),
) raises:
    """Write N models into ONE v3 checkpoint file: a single header, then each
    model's Param + State sections, in pack order. `load_params_multi` walks the
    same models in the same order, so each section's dotted name is validated
    against its own model — duplicate names across models never collide. Replaces
    the per-model sidecar layout (plain `save_params` is whole-file-per-model)."""
    var w = BinaryCheckpointWriter(save_moments)

    comptime for i in range(models.__len__()):
        w.mode = 0
        walk_params[target](models[i], w, ctx)
        w.mode = 1
        var _sref4 = ParamVisitorRef.of[type_of(w), target](w)
        models[i].for_each_state[target](_sref4, ctx)
    w.write_scalars(scalars)
    _write_file_bytes(path, w.content)


def load_params_multi[
    target: StaticString, *Ms: ParamWalkable
](
    path: String,
    ctx: Optional[DeviceContext],
    mut *models: *Ms,
) raises -> CheckpointScalars:
    """Load a single-file multi-model checkpoint written by `save_params_multi`
    (v3 binary, or legacy v2 text), walking the models in the same pack order
    and validating each one's names/sizes against the file's sections. Returns
    the file's `K` scalars (empty for v2)."""
    var bytes = _read_file_bytes(path)
    if _is_v3_header(bytes):
        var rb = BinaryCheckpointReader(bytes^)
        comptime for i in range(models.__len__()):
            rb.mode = 0
            walk_params[target](models[i], rb, ctx)
            rb.mode = 1
            var _sref5 = ParamVisitorRef.of[type_of(rb), target](rb)
            models[i].for_each_state[target](_sref5, ctx)
        # ⚠ `finish` is also the mention that keeps `rb` alive through the last
        # state pass — see `BinaryCheckpointReader.finish`. This path had no
        # mention after the loop, so the last model's state pass walked a
        # reader that was already destroyed.
        var sc = rb.read_scalars()
        rb.finish()
        return sc^
    var content: String
    with open(path, "r") as f:
        content = String(f.read())
    var lines = _split_lines(content)
    var body = List[String]()
    for li in range(len(lines)):
        if lines[li].startswith("storage-ckpt"):
            continue
        body.append(lines[li])
    var r = CheckpointReader(body^)

    comptime for i in range(models.__len__()):
        r.mode = 0
        walk_params[target](models[i], r, ctx)
        r.mode = 1
        var _sref6 = ParamVisitorRef.of[type_of(r), target](r)
        models[i].for_each_state[target](_sref6, ctx)
        _ = r.cur  # keep `r` alive past the pointer hand-off; see finish()
    return CheckpointScalars()
