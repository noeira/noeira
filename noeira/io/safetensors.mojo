# +--------------------------------------------------------------------------+ #
# | safetensors — a length, a JSON header, and a blob
# +--------------------------------------------------------------------------+ #
"""Read and write `.safetensors` without Python.

    from noeira.io.safetensors import SafeTensors, SafeTensorsWriter

    var f = SafeTensors(String("model.safetensors"))
    var w = f.read_f32(String("layer4.1.conv2.weight"))   # 512*512*3*3 values

## The format, in full

    [8 bytes LE u64: header length N][N bytes UTF-8 JSON][the tensor bytes]

and the JSON is one object mapping a tensor name to
`{"dtype": "F32", "shape": [512,512,3,3], "data_offsets": [begin, end]}`,
where the offsets are relative to the START OF THE BLOB, not of the file. An
optional `"__metadata__"` member holds a flat string -> string map and is the
only place a producer may put anything else. There is no compression, no
framing, no versioning and no nesting: the format exists because the
alternative was pickle, which executes code on load.

That is the whole specification. It is short enough that the interesting part
of this file is the validation, not the parsing.

## What is supported

Reading: `F32`, `F64`, `F16`, `BF16` — the last three widen to f32 on the way
in, because this framework is f32 everywhere (`nn.constants.DT`). Every other
dtype is REJECTED BY NAME rather than silently reinterpreted; integer tensors
do occur in real files (`num_batches_tracked` is `I64` with shape `[]`) and a
caller that means to skip them should check `dtype_of` and skip them, not
receive garbage floats. Writing: `F32`, and `BF16` rounded to nearest-even
from f32 values, as torch's `.to(torch.bfloat16)` rounds.

⚠ Host byte order is assumed little-endian, which every platform this repo
builds for is. The dtype conversions read bytes explicitly, so only the raw
`F32`/`F64` `memcpy` paths carry the assumption.

## Four things that bite

* **The header's key order is not the walk order.** Producers commonly sort
  alphabetically, which is why a torchvision ResNet18 file begins with
  `bn1.num_batches_tracked` rather than `conv1.weight`. Anything that loads by
  position rather than by name will read the right sizes into the wrong
  tensors, on a file that is entirely valid.
* **The header length is untrusted input.** It is 8 attacker-controlled bytes
  in front of an allocation, which is where safetensors' own CVEs lived. It is
  bounded against the file size AND against `MAX_HEADER_BYTES` before anything
  is allocated.
* **The offsets are untrusted too.** Every `[begin, end]` is checked to lie
  inside the blob and to match `numel * itemsize` for the declared shape, and
  the entries as a whole are checked to TILE the blob exactly — no gaps, no
  overlaps. A file that fails the tiling check is one the reference
  implementation rejects as well (`MetadataIncompleteBuffer`), so accepting it
  would make us disagree with every other reader.
* **Nothing here is lazy about correctness but reading IS lazy about bytes.**
  The header is parsed once; each tensor is `seek`+`read` on demand. A 46 MB
  ResNet18 could be resident, a 30 GB shard could not, and the difference
  should not be a rewrite.

## Gating

`tests/io/test_safetensors.mojo` runs against files written by the REFERENCE
implementation (`tools/nn/dump_safetensors_reference.py`, `-e act-ref`), and
that script also reads OUR output back with the reference library. Both
directions are checked against the library everyone else uses, deliberately
rather than against a second Python transcription of this same understanding:
two parsers sharing one wrong assumption is a gate that cannot see the
assumption.
"""

from std.memory import bitcast, unsafe_memcpy

from .fileio import file_size, read_file_range, write_file_atomic
from .json import J_ARRAY, J_OBJECT, JsonDoc, kind_name, parse_json


# ══════════════════════════════════════════════════════════════════════════
# dtypes
# ══════════════════════════════════════════════════════════════════════════

comptime ST_UNKNOWN = -1
comptime ST_BOOL = 0
comptime ST_U8 = 1
comptime ST_I8 = 2
comptime ST_F8_E5M2 = 3
comptime ST_F8_E4M3 = 4
comptime ST_I16 = 5
comptime ST_U16 = 6
comptime ST_F16 = 7
comptime ST_BF16 = 8
comptime ST_I32 = 9
comptime ST_U32 = 10
comptime ST_F32 = 11
comptime ST_I64 = 12
comptime ST_U64 = 13
comptime ST_F64 = 14

comptime MAX_HEADER_BYTES = 100 * 1024 * 1024
"""A header is metadata about tensors; 100 MB of it is a malformed or hostile
file, not a big model. Bounded BEFORE the allocation, not after."""

comptime MAX_NUMEL = 1 << 44
"""Guards `numel * itemsize` against overflowing Int on a crafted shape."""


def dtype_code(name: String) -> Int:
    """The safetensors dtype string -> our code, or `ST_UNKNOWN`."""
    if name == "F32": return ST_F32
    if name == "F64": return ST_F64
    if name == "F16": return ST_F16
    if name == "BF16": return ST_BF16
    if name == "I64": return ST_I64
    if name == "U64": return ST_U64
    if name == "I32": return ST_I32
    if name == "U32": return ST_U32
    if name == "I16": return ST_I16
    if name == "U16": return ST_U16
    if name == "I8": return ST_I8
    if name == "U8": return ST_U8
    if name == "BOOL": return ST_BOOL
    if name == "F8_E5M2": return ST_F8_E5M2
    if name == "F8_E4M3": return ST_F8_E4M3
    return ST_UNKNOWN


def dtype_name(code: Int) -> String:
    if code == ST_F32: return String("F32")
    if code == ST_F64: return String("F64")
    if code == ST_F16: return String("F16")
    if code == ST_BF16: return String("BF16")
    if code == ST_I64: return String("I64")
    if code == ST_U64: return String("U64")
    if code == ST_I32: return String("I32")
    if code == ST_U32: return String("U32")
    if code == ST_I16: return String("I16")
    if code == ST_U16: return String("U16")
    if code == ST_I8: return String("I8")
    if code == ST_U8: return String("U8")
    if code == ST_BOOL: return String("BOOL")
    if code == ST_F8_E5M2: return String("F8_E5M2")
    if code == ST_F8_E4M3: return String("F8_E4M3")
    return String("<unknown>")


def dtype_size(code: Int) -> Int:
    """Bytes per element. 0 for an unknown code."""
    if code == ST_F64 or code == ST_I64 or code == ST_U64: return 8
    if code == ST_F32 or code == ST_I32 or code == ST_U32: return 4
    if code == ST_F16 or code == ST_BF16 or code == ST_I16 or code == ST_U16:
        return 2
    if code == ST_I8 or code == ST_U8 or code == ST_BOOL: return 1
    if code == ST_F8_E5M2 or code == ST_F8_E4M3: return 1
    return 0


def is_float_dtype(code: Int) -> Bool:
    """Whether `read_f32` can widen this dtype."""
    return (
        code == ST_F32 or code == ST_F64 or code == ST_F16 or code == ST_BF16
    )


# ══════════════════════════════════════════════════════════════════════════
# Reading
# ══════════════════════════════════════════════════════════════════════════


def _le_u64(ref b: List[UInt8], off: Int) -> UInt64:
    var v = UInt64(0)
    for k in range(8):
        v |= UInt64(Int(b[off + k])) << UInt64(8 * k)
    return v


struct SafeTensors(Movable):
    """A parsed `.safetensors` header. Tensor bytes are read on demand.

    Parallel arrays rather than a `List[STEntry]`: shapes are variable-length,
    and one flat `shape_data` with per-entry (start, rank) avoids a nested
    `List[List[Int]]` and its copy semantics.
    """

    var path: String
    var names: List[String]
    var dtypes: List[Int]
    var begins: List[Int]
    """Relative to the start of the blob, as the file states them."""
    var ends: List[Int]
    var shape_data: List[Int]
    var shape_start: List[Int]
    var shape_rank: List[Int]
    var data_start: Int
    """ABSOLUTE file offset of the blob, i.e. `8 + header_len`."""
    var data_len: Int
    var meta_keys: List[String]
    var meta_vals: List[String]

    def __init__(out self, var path: String) raises:
        self.path = path^
        self.names = List[String]()
        self.dtypes = List[Int]()
        self.begins = List[Int]()
        self.ends = List[Int]()
        self.shape_data = List[Int]()
        self.shape_start = List[Int]()
        self.shape_rank = List[Int]()
        self.meta_keys = List[String]()
        self.meta_vals = List[String]()

        var fsz = file_size(self.path)
        if fsz < 8:
            raise Error(
                "safetensors: '" + self.path + "' is " + String(fsz)
                + " bytes — too short to hold even the header length"
            )

        var head8 = read_file_range(self.path, 0, 8)
        var hlen64 = _le_u64(head8, 0)
        # ⚠ Bound BEFORE narrowing to Int and before allocating. A u64 of
        # 0xFFFFFFFFFFFFFFFF narrows to -1, and -1 is a fine-looking length
        # right up until it is used as one.
        if hlen64 > UInt64(fsz - 8):
            raise Error(
                "safetensors: '" + self.path + "' declares a "
                + String(hlen64) + "-byte header but holds only "
                + String(fsz - 8) + " bytes after the length prefix"
            )
        if hlen64 > UInt64(MAX_HEADER_BYTES):
            raise Error(
                "safetensors: '" + self.path + "' declares a "
                + String(hlen64) + "-byte header, over this reader's "
                + String(MAX_HEADER_BYTES) + "-byte limit"
            )
        var hlen = Int(hlen64)
        self.data_start = 8 + hlen
        self.data_len = fsz - self.data_start

        var doc = parse_json(read_file_range(self.path, 8, hlen))
        var root = doc.root()
        if doc.kind_of(root) != J_OBJECT:
            raise Error(
                "safetensors: '" + self.path + "' header is a JSON "
                + kind_name(doc.kind_of(root)) + ", not an object"
            )

        for i in range(doc.size(root)):
            var key = doc.key_at(root, i)
            var val = doc.at(root, i)
            if key == "__metadata__":
                if doc.kind_of(val) != J_OBJECT:
                    raise Error(
                        "safetensors: '" + self.path + "' __metadata__ is not"
                        " an object"
                    )
                for j in range(doc.size(val)):
                    self.meta_keys.append(doc.key_at(val, j))
                    self.meta_vals.append(doc.string(doc.at(val, j)))
                continue
            self._add_entry(doc, key, val)

        self._validate()

    def __init__(out self, *, deinit move: Self):
        self.path = move.path^
        self.names = move.names^
        self.dtypes = move.dtypes^
        self.begins = move.begins^
        self.ends = move.ends^
        self.shape_data = move.shape_data^
        self.shape_start = move.shape_start^
        self.shape_rank = move.shape_rank^
        self.data_start = move.data_start
        self.data_len = move.data_len
        self.meta_keys = move.meta_keys^
        self.meta_vals = move.meta_vals^

    def _add_entry(
        mut self, ref doc: JsonDoc, var key: String, val: Int
    ) raises:
        if doc.kind_of(val) != J_OBJECT:
            raise Error(
                "safetensors: entry '" + key + "' is not an object"
            )
        var dt_node = doc.field(val, String("dtype"))
        var sh_node = doc.field(val, String("shape"))
        var of_node = doc.field(val, String("data_offsets"))
        if dt_node < 0 or sh_node < 0 or of_node < 0:
            raise Error(
                "safetensors: entry '" + key + "' is missing dtype, shape or"
                " data_offsets"
            )
        var dt = dtype_code(doc.string(dt_node))
        if dt == ST_UNKNOWN:
            raise Error(
                "safetensors: entry '" + key + "' has dtype '"
                + doc.string(dt_node) + "', which is not in the specification"
            )
        if doc.kind_of(sh_node) != J_ARRAY:
            raise Error("safetensors: entry '" + key + "' shape is not a list")
        if doc.kind_of(of_node) != J_ARRAY or doc.size(of_node) != 2:
            raise Error(
                "safetensors: entry '" + key + "' data_offsets is not a"
                " 2-element list"
            )

        var start = len(self.shape_data)
        var numel = 1
        for d in range(doc.size(sh_node)):
            var dim = doc.integer(doc.at(sh_node, d))
            if dim < 0:
                raise Error(
                    "safetensors: entry '" + key + "' has a negative dimension "
                    + String(dim)
                )
            self.shape_data.append(dim)
            if dim != 0 and numel > MAX_NUMEL // dim:
                raise Error(
                    "safetensors: entry '" + key + "' declares more than "
                    + String(MAX_NUMEL) + " elements"
                )
            numel *= dim

        var begin = doc.integer(doc.at(of_node, 0))
        var end = doc.integer(doc.at(of_node, 1))
        if begin < 0 or end < begin or end > self.data_len:
            raise Error(
                "safetensors: entry '" + key + "' spans [" + String(begin)
                + ", " + String(end) + ") outside the "
                + String(self.data_len) + "-byte data block"
            )
        var want = numel * dtype_size(dt)
        if end - begin != want:
            raise Error(
                "safetensors: entry '" + key + "' is " + String(end - begin)
                + " bytes but its shape and dtype (" + dtype_name(dt)
                + ") need " + String(want)
            )

        self.names.append(key^)
        self.dtypes.append(dt)
        self.shape_start.append(start)
        self.shape_rank.append(len(self.shape_data) - start)
        self.begins.append(begin)
        self.ends.append(end)

    def _validate(self) raises:
        """The entries must TILE the blob: sorted by offset, each one starts
        where the previous ended, and the last ends at the blob's end.

        Per-entry bounds (checked in `_add_entry`) already make every read
        safe. This catches the different failure — a file whose offsets are
        individually plausible and collectively wrong, which every other
        reader rejects and which we would otherwise load as silently shifted
        weights."""
        var n = len(self.begins)
        if n == 0:
            if self.data_len != 0:
                raise Error(
                    "safetensors: '" + self.path + "' names no tensors but"
                    " carries " + String(self.data_len) + " bytes of data"
                )
            return

        var order = _argsort(self.begins)
        var cursor = 0
        for k in range(n):
            var i = order[k]
            if self.begins[i] != cursor:
                raise Error(
                    "safetensors: '" + self.path + "' has "
                    + ("a gap" if self.begins[i] > cursor else "an overlap")
                    + " at byte " + String(cursor) + ": '" + self.names[i]
                    + "' starts at " + String(self.begins[i])
                )
            cursor = self.ends[i]
        if cursor != self.data_len:
            raise Error(
                "safetensors: '" + self.path + "' tensors cover "
                + String(cursor) + " bytes but the data block is "
                + String(self.data_len)
            )
        _check_unique(self.names, self.path)

    # ── lookup ────────────────────────────────────────────────────────────

    def size(self) -> Int:
        return len(self.names)

    def index(self, name: String) -> Int:
        """The entry's position, or -1."""
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def has(self, name: String) -> Bool:
        return self.index(name) >= 0

    def _need(self, name: String) raises -> Int:
        var i = self.index(name)
        if i < 0:
            raise Error(
                "safetensors: '" + self.path + "' has no tensor named '"
                + name + "'"
            )
        return i

    def dtype_of(self, name: String) raises -> Int:
        return self.dtypes[self._need(name)]

    def shape(self, name: String) raises -> List[Int]:
        var i = self._need(name)
        var out = List[Int]()
        for k in range(self.shape_rank[i]):
            out.append(self.shape_data[self.shape_start[i] + k])
        return out^

    def numel(self, name: String) raises -> Int:
        var i = self._need(name)
        var n = 1
        for k in range(self.shape_rank[i]):
            n *= self.shape_data[self.shape_start[i] + k]
        return n

    def metadata(self, key: String) -> String:
        """The `__metadata__` value for `key`, or "" — the format gives no way
        to distinguish an absent key from an empty one."""
        for i in range(len(self.meta_keys)):
            if self.meta_keys[i] == key:
                return String(self.meta_vals[i])
        return String("")

    def shape_str(self, name: String) raises -> String:
        var s = self.shape(name)
        var out = String("[")
        for i in range(len(s)):
            if i > 0:
                out += ", "
            out += String(s[i])
        return out + "]"

    # ── values ────────────────────────────────────────────────────────────

    def read_raw(self, name: String) raises -> List[UInt8]:
        var i = self._need(name)
        return read_file_range(
            self.path,
            self.data_start + self.begins[i],
            self.ends[i] - self.begins[i],
        )

    def read_f32(self, name: String) raises -> List[Float32]:
        """Values as f32, widening F16/BF16/F64. Raises BY NAME on an integer
        or 8-bit-float dtype rather than reinterpreting its bits."""
        var i = self._need(name)
        var dt = self.dtypes[i]
        if not is_float_dtype(dt):
            raise Error(
                "safetensors: '" + name + "' has dtype " + dtype_name(dt)
                + "; read_f32 handles F32/F64/F16/BF16 only (check `dtype_of`"
                " and skip it, or use `read_raw`)"
            )
        var n = self.numel(name)
        var raw = self.read_raw(name)
        var out = List[Float32](unsafe_uninit_length=n)
        if n == 0:
            return out^

        if dt == ST_F32:
            # memcpy rather than a bitcast load: `List[UInt8]`'s buffer carries
            # no 4-byte alignment guarantee, and an unaligned typed load is UB
            # even where the hardware tolerates it.
            unsafe_memcpy(
                dest=out.unsafe_ptr().unsafe_bitcast[UInt8](),
                src=raw.unsafe_ptr(),
                count=n * 4,
            )
        elif dt == ST_F64:
            var tmp = List[Float64](unsafe_uninit_length=n)
            unsafe_memcpy(
                dest=tmp.unsafe_ptr().unsafe_bitcast[UInt8](),
                src=raw.unsafe_ptr(),
                count=n * 8,
            )
            for k in range(n):
                out[k] = Float32(tmp[k])
        elif dt == ST_BF16:
            # bfloat16 IS the top half of an f32 — the widening is a shift,
            # exactly, with no rounding and no special-case for inf/nan.
            for k in range(n):
                var lo = UInt32(Int(raw[2 * k]))
                var hi = UInt32(Int(raw[2 * k + 1]))
                var bits = ((hi << UInt32(8)) | lo) << UInt32(16)
                out[k] = bitcast[DType.float32](bits)
        else:  # ST_F16
            for k in range(n):
                var lo = UInt16(Int(raw[2 * k]))
                var hi = UInt16(Int(raw[2 * k + 1]))
                var h = bitcast[DType.float16]((hi << UInt16(8)) | lo)
                out[k] = h.cast[DType.float32]()
        return out^


def _argsort(ref xs: List[Int]) -> List[Int]:
    """Indices of `xs` in ascending value order. Bottom-up merge sort: a real
    file can carry thousands of tensors and an insertion sort's quadratic term
    would show up as a mysterious pause on load."""
    var n = len(xs)
    var a = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        a[i] = i
    if n < 2:
        return a^
    var b = List[Int](unsafe_uninit_length=n)
    var width = 1
    while width < n:
        var i = 0
        while i < n:
            var mid = i + width
            if mid > n:
                mid = n
            var hi = i + 2 * width
            if hi > n:
                hi = n
            var l = i
            var r = mid
            var k = i
            while l < mid or r < hi:
                if r >= hi or (l < mid and xs[a[l]] <= xs[a[r]]):
                    b[k] = a[l]
                    l += 1
                else:
                    b[k] = a[r]
                    r += 1
                k += 1
            i = hi
        for j in range(n):
            a[j] = b[j]
        width *= 2
    return a^


def _check_unique(ref names: List[String], path: String) raises:
    """A duplicate key is legal JSON and illegal here — our `field` and `index`
    lookups take the LAST and the FIRST respectively, so a duplicated name
    makes the file mean two different things depending on who asked.

    ⚠ The tiling check does NOT subsume this. Two entries under one name with
    adjacent, non-overlapping offsets tile the blob perfectly; nothing about
    the byte layout is wrong. Only the name is.

    Sorted by hash, then compared for real only within a hash collision: the
    obvious nested loop is quadratic in the tensor count, and a large model can
    carry thousands."""
    var n = len(names)
    if n < 2:
        return
    var hashes = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        hashes[i] = Int(hash(names[i]))
    var order = _argsort(hashes)
    for k in range(1, n):
        var a = order[k - 1]
        var b = order[k]
        if hashes[a] == hashes[b] and names[a] == names[b]:
            raise Error(
                "safetensors: '" + path + "' names '" + names[a] + "' twice"
            )


# ══════════════════════════════════════════════════════════════════════════
# Writing
# ══════════════════════════════════════════════════════════════════════════


def f32_to_bf16_bits(v: Float32) -> UInt16:
    """The bfloat16 bits of `v`, rounded to nearest with ties to even, the
    rounding torch's `.to(torch.bfloat16)` does (c10's
    `round_to_nearest_even`): add 0x7FFF plus the lowest kept bit, keep the
    top 16. A NaN becomes torch's quiet NaN, 0x7FC0, which the add could
    otherwise carry into an infinity; ±inf and overflow come out as ±inf."""
    var bits = bitcast[DType.uint32](v)
    if (bits & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
        return UInt16(0x7FC0)
    var bias = UInt32(0x7FFF) + ((bits >> UInt32(16)) & UInt32(1))
    return UInt16((bits + bias) >> UInt32(16))


def _json_escape(s: String) -> String:
    """JSON string body for `s`. Tensor names are dotted ASCII in practice, so
    this rarely does anything — which is the reason to write it now rather
    than discover the one name that needed it from a corrupt file."""
    var out = String("")
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c == ord('"'):
            out += '\\"'
        elif c == ord("\\"):
            out += "\\\\"
        elif c == 0x08:
            out += "\\b"
        elif c == 0x0C:
            out += "\\f"
        elif c == 0x0A:
            out += "\\n"
        elif c == 0x0D:
            out += "\\r"
        elif c == 0x09:
            out += "\\t"
        elif c < 0x20:
            var hexd = String("0123456789abcdef")
            out += "\\u00"
            out += hexd[byte = (c >> 4) & 0xF]
            out += hexd[byte = c & 0xF]
        else:
            # >= 0x80 bytes pass through: they are already the UTF-8 the
            # header is defined to be.
            out += chr(c)
    return out^


struct SafeTensorsWriter(Movable):
    """Accumulate named f32 tensors, then write one `.safetensors`.

    The blob is built in memory, so peak cost is the model's size once — the
    same as `nn/core/checkpoint.mojo`'s writer, and the reason neither is the
    right tool for a model that does not fit in RAM.
    """

    var names: List[String]
    var shape_data: List[Int]
    var shape_start: List[Int]
    var shape_rank: List[Int]
    var begins: List[Int]
    var ends: List[Int]
    var dtypes: List[Int]
    """Per tensor, the header's dtype: `ST_F32`, `ST_BF16` or `ST_U8`."""
    var data: List[UInt8]
    var meta_keys: List[String]
    var meta_vals: List[String]

    def __init__(out self):
        self.names = List[String]()
        self.shape_data = List[Int]()
        self.shape_start = List[Int]()
        self.shape_rank = List[Int]()
        self.begins = List[Int]()
        self.ends = List[Int]()
        self.dtypes = List[Int]()
        self.data = List[UInt8]()
        self.meta_keys = List[String]()
        self.meta_vals = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.names = move.names^
        self.shape_data = move.shape_data^
        self.shape_start = move.shape_start^
        self.shape_rank = move.shape_rank^
        self.begins = move.begins^
        self.ends = move.ends^
        self.dtypes = move.dtypes^
        self.data = move.data^
        self.meta_keys = move.meta_keys^
        self.meta_vals = move.meta_vals^

    def add_metadata(mut self, var key: String, var value: String):
        self.meta_keys.append(key^)
        self.meta_vals.append(value^)

    def size(self) -> Int:
        return len(self.names)

    def _entry(
        mut self, name: String, ref shape: List[Int], count: Int, n_vals: Int
    ) raises:
        """Validate one tensor and record its shape. `add_f32` and `add_bf16`
        share it, then append the name, dtype and bytes themselves.

        ⚠ `shape` is CHECKED against `count`, not derived from it. The whole
        point of carrying a shape is that a consumer reshapes by it; a shape
        that does not match the data would produce a file that loads and is
        wrong."""
        for i in range(len(self.names)):
            if self.names[i] == name:
                raise Error(
                    "safetensors: '" + name + "' added twice — the header is"
                    " a JSON object and the second one would shadow the first"
                )
        var numel = 1
        for i in range(len(shape)):
            if shape[i] < 0:
                raise Error(
                    "safetensors: '" + name + "' has a negative dimension "
                    + String(shape[i])
                )
            numel *= shape[i]
        if numel != count:
            var s = String("[")
            for i in range(len(shape)):
                if i > 0:
                    s += ", "
                s += String(shape[i])
            raise Error(
                "safetensors: '" + name + "' declares shape " + s + "] = "
                + String(numel) + " elements but carries " + String(count)
            )
        if count < 0 or count > n_vals:
            raise Error(
                "safetensors: '" + name + "' asks for " + String(count)
                + " values from a list of " + String(n_vals)
            )
        var start = len(self.shape_data)
        for i in range(len(shape)):
            self.shape_data.append(shape[i])
        self.shape_start.append(start)
        self.shape_rank.append(len(shape))

    def add_f32(
        mut self,
        var name: String,
        ref shape: List[Int],
        ref vals: List[Float32],
        count: Int,
    ) raises:
        """Append the FIRST `count` values of `vals` under `name`, with the
        declared `shape`.

        ⚠ `count` is separate from `len(vals)` on purpose: an `nn` `Tensor`'s
        backing list can be longer than the param it holds (padded or reused),
        so the walk's comptime `N` is the authority on how much is live."""
        self._entry(name, shape, count, len(vals))
        var begin = len(self.data)
        self.data.resize(begin + count * 4, 0)
        unsafe_memcpy(
            dest=self.data.unsafe_ptr().unsafe_offset(begin),
            src=vals.unsafe_ptr().unsafe_bitcast[UInt8](),
            count=count * 4,
        )
        self.names.append(name^)
        self.dtypes.append(ST_F32)
        self.begins.append(begin)
        self.ends.append(begin + count * 4)

    def add_bf16(
        mut self,
        var name: String,
        ref shape: List[Int],
        ref vals: List[Float32],
        count: Int,
    ) raises:
        """`add_f32`, stored as bfloat16: each value rounded to nearest, ties
        to even (`f32_to_bf16_bits`). For a file a bf16 consumer reads, such
        as a PEFT adapter for a bf16 model."""
        self._entry(name, shape, count, len(vals))
        var begin = len(self.data)
        self.data.resize(begin + count * 2, 0)
        for k in range(count):
            var b = f32_to_bf16_bits(vals[k])
            self.data[begin + 2 * k] = UInt8(Int(b & 0xFF))
            self.data[begin + 2 * k + 1] = UInt8(Int(b >> 8))
        self.names.append(name^)
        self.dtypes.append(ST_BF16)
        self.begins.append(begin)
        self.ends.append(begin + count * 2)

    def add_u8(
        mut self,
        var name: String,
        ref shape: List[Int],
        ref vals: List[UInt8],
        count: Int,
    ) raises:
        """Append the first `count` bytes of `vals` under `name` as U8, with
        the declared `shape`. For quantized weights, whose blocks are bytes
        (`nn/quant/q4_0.mojo`); `SafeTensors.read_raw` reads them back."""
        self._entry(name, shape, count, len(vals))
        var begin = len(self.data)
        self.data.resize(begin + count, 0)
        if count > 0:
            unsafe_memcpy(
                dest=self.data.unsafe_ptr().unsafe_offset(begin),
                src=vals.unsafe_ptr(),
                count=count,
            )
        self.names.append(name^)
        self.dtypes.append(ST_U8)
        self.begins.append(begin)
        self.ends.append(begin + count)

    def add_f32_list(
        mut self, var name: String, ref shape: List[Int], ref vals: List[Float32]
    ) raises:
        self.add_f32(name^, shape, vals, len(vals))

    def header(self) -> String:
        """The JSON header, unpadded. Separate from `save` so a gate can read
        it without a file."""
        var out = String("{")
        var first = True
        if len(self.meta_keys) > 0:
            out += '"__metadata__":{'
            for i in range(len(self.meta_keys)):
                if i > 0:
                    out += ","
                out += '"' + _json_escape(self.meta_keys[i]) + '":"'
                out += _json_escape(self.meta_vals[i]) + '"'
            out += "}"
            first = False
        for i in range(len(self.names)):
            if not first:
                out += ","
            first = False
            out += '"' + _json_escape(self.names[i]) + '":{"dtype":"'
            out += dtype_name(self.dtypes[i]) + '"'
            out += ',"shape":['
            for k in range(self.shape_rank[i]):
                if k > 0:
                    out += ","
                out += String(self.shape_data[self.shape_start[i] + k])
            out += '],"data_offsets":[' + String(self.begins[i]) + ","
            out += String(self.ends[i]) + "]}"
        return out + "}"

    def save(self, var path: String) raises:
        """Write the file. Tensors appear in the order they were added, which
        for a model walk is the module order — legal, and more useful to a
        human than the alphabetical order most producers emit."""
        var head = self.header()
        # Pad the header with spaces so the blob starts 8-byte aligned. Not
        # required by the format (real files in the wild are unaligned) but
        # free, and it lets a consumer mmap the blob and cast it.
        var hb = head.as_bytes()
        var hlen = len(hb)
        var pad = (8 - ((8 + hlen) % 8)) % 8

        var buf = List[UInt8]()
        buf.reserve(8 + hlen + pad + len(self.data))
        var total = UInt64(hlen + pad)
        for k in range(8):
            buf.append(UInt8(Int((total >> UInt64(8 * k)) & UInt64(0xFF))))
        for i in range(hlen):
            buf.append(hb[i])
        for _ in range(pad):
            buf.append(UInt8(0x20))
        var off = len(buf)
        buf.resize(off + len(self.data), 0)
        if len(self.data) > 0:
            unsafe_memcpy(
                dest=buf.unsafe_ptr().unsafe_offset(off),
                src=self.data.unsafe_ptr(),
                count=len(self.data),
            )
        write_file_atomic(path^, buf)
