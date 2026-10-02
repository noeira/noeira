# +--------------------------------------------------------------------------+ #
# | NumPy `.npy` v1/v2 reader — the one format AMASS ships in
# +--------------------------------------------------------------------------+ #
"""Read a dense `.npy` array into a `List[Float64]`, no Python.

`fleaven/Retargeted_AMASS_for_robotics` stores one `[N, 36]` float64 array
per clip (`noeira/data/amass.mojo`), so this reader exists for the same
reason `io/pickle.mojo` does: the file format is between us and the data,
and importing numpy to cross it would put Python back on the data path.

THE FORMAT (numpy's `lib/format.py`, the only normative description):

    \\x93NUMPY  <major:u8> <minor:u8>  <hlen>  <header…>  <raw data>

`hlen` is u16 little-endian at v1.0 and u32 at v2.0/v3.0; the header is an
ASCII Python dict literal padded with spaces so that data starts aligned to
64 bytes, and it ends with `\\n`. Three keys matter: `descr` (a dtype
string), `fortran_order` (a bool) and `shape` (a tuple).

WHAT IS SUPPORTED, AND WHY THE REST RAISES. `<f8`, `<f4`, `<i8`, `<i4` and
their `=`/`|`-prefixed and prefix-less spellings, C order only. Everything
else raises by NAME:

  * big-endian (`>f8`) would need a byte swap that no file we read wants,
    and silently not swapping reads garbage that looks like data;
  * `fortran_order: True` would transpose the array, and a transposed
    `[N, 36]` is `[36, N]` — the same bytes, a different meaning, which is
    exactly the mistake a tolerant reader makes for you;
  * object arrays (`|O`) are pickles, which is `io/pickle.mojo`'s job.

⚠ THE HEADER IS PARSED, NOT PATTERN-MATCHED. numpy writes `'descr': '<f8'`
with single quotes and one space, but the format only promises a dict
literal — `repr` of a dict whose key order and spacing numpy has changed
across releases (v1.0 wrote `(360, 36)`, and a 1-D shape is `(360,)` with
the trailing comma). `_find_key` scans for the key, skips to the value and
reads it, so spacing and ordering do not matter and a 1-D shape is not a
syntax error.
"""

from std.os.path import exists

from noeira.core.bytes import string_from_byte_span
from noeira.io.fileio import read_file_bytes, read_file_range, file_size


comptime NPY_MAGIC_LEN: Int = 8          # b"\x93NUMPY" + major + minor


struct NpyHeader(Copyable, Movable):
    """What the header says, and where the data starts."""
    var dtype: String            # the normalised `descr`, e.g. "<f8"
    var item_size: Int           # bytes per element
    var is_float: Bool
    var rows: Int                # shape[0], or 1 for a 0-D array
    var cols: Int                # the product of shape[1:], or 1
    var data_offset: Int
    var n_elems: Int

    def __init__(
        out self, var dtype: String, item_size: Int, is_float: Bool,
        rows: Int, cols: Int, data_offset: Int, n_elems: Int,
    ):
        self.dtype = dtype^
        self.item_size = item_size
        self.is_float = is_float
        self.rows = rows
        self.cols = cols
        self.data_offset = data_offset
        self.n_elems = n_elems


@always_inline
def _is_space(c: UInt8) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13


def _find_key(h: List[UInt8], key: String) raises -> Int:
    """The offset just past `'<key>':` in the header dict, or −1.

    Matches the key inside either quote style and tolerates any spacing
    around the colon — the format promises a dict literal, not a layout.
    """
    var kb = key.as_bytes()
    var kn = key.byte_length()
    for i in range(len(h)):
        if h[i] != 39 and h[i] != 34:        # ' or "
            continue
        if i + 1 + kn + 1 > len(h):
            break
        var hit = True
        for j in range(kn):
            if h[i + 1 + j] != kb[j]:
                hit = False
                break
        if not hit or h[i + 1 + kn] != h[i]:  # the closing quote must match
            continue
        var p = i + kn + 2
        while p < len(h) and _is_space(h[p]):
            p += 1
        if p >= len(h) or h[p] != 58:          # :
            continue
        p += 1
        while p < len(h) and _is_space(h[p]):
            p += 1
        return p
    return -1


def _read_quoted(h: List[UInt8], start: Int) raises -> String:
    """The string literal at `start`."""
    if start >= len(h) or (h[start] != 39 and h[start] != 34):
        raise Error("npy: a quoted value was expected in the header")
    var q = h[start]
    var e = start + 1
    while e < len(h) and h[e] != q:
        e += 1
    if e >= len(h):
        raise Error("npy: an unterminated string in the header")
    return string_from_byte_span(h, start + 1, e)


def _read_tuple(h: List[UInt8], start: Int) raises -> List[Int]:
    """The integer tuple at `start`. `()` is empty, `(360,)` is one element."""
    if start >= len(h) or h[start] != 40:      # (
        raise Error("npy: `shape` is not a tuple")
    var out = List[Int]()
    var p = start + 1
    var cur = 0
    var have = False
    while p < len(h):
        var c = h[p]
        if c >= 48 and c <= 57:
            cur = cur * 10 + Int(c - 48)
            have = True
        elif c == 44 or c == 41:               # , or )
            if have:
                out.append(cur)
            cur = 0
            have = False
            if c == 41:
                return out^
        elif _is_space(c):
            pass
        else:
            raise Error(
                "npy: `shape` holds a non-integer at byte " + String(p)
            )
        p += 1
    raise Error("npy: an unterminated `shape` tuple")


def parse_npy_header(prefix: List[UInt8], total_size: Int) raises -> NpyHeader:
    """Parse the magic + header from the first bytes of a `.npy` file."""
    if len(prefix) < NPY_MAGIC_LEN + 2:
        raise Error(
            "npy: the file is " + String(total_size)
            + " bytes — too short to hold a header"
        )
    if (prefix[0] != 0x93 or prefix[1] != 78 or prefix[2] != 85
            or prefix[3] != 77 or prefix[4] != 80 or prefix[5] != 89):
        raise Error("npy: the magic is not \\x93NUMPY — this is not a .npy file")
    var major = Int(prefix[6])
    var hlen: Int
    var hstart: Int
    if major == 1:
        hlen = Int(prefix[8]) | (Int(prefix[9]) << 8)
        hstart = NPY_MAGIC_LEN + 2
    elif major == 2 or major == 3:
        if len(prefix) < NPY_MAGIC_LEN + 4:
            raise Error("npy: a v" + String(major) + " header was truncated")
        hlen = (
            Int(prefix[8]) | (Int(prefix[9]) << 8)
            | (Int(prefix[10]) << 16) | (Int(prefix[11]) << 24)
        )
        hstart = NPY_MAGIC_LEN + 4
    else:
        raise Error("npy: version " + String(major) + " is not one we read")
    if hstart + hlen > len(prefix):
        raise Error(
            "npy: the header claims " + String(hlen) + " bytes but only "
            + String(len(prefix) - hstart) + " were read"
        )

    var h = List[UInt8]()
    for i in range(hstart, hstart + hlen):
        h.append(prefix[i])

    # ── descr ────────────────────────────────────────────────────────────
    var kd = _find_key(h, String("descr"))
    if kd < 0:
        raise Error("npy: the header has no `descr`")
    var descr = _read_quoted(h, kd)
    var endian = String("<")
    var body = descr
    if descr.byte_length() >= 1:
        var c0 = String(descr[byte=0:1])
        if c0 == "<" or c0 == ">" or c0 == "=" or c0 == "|":
            endian = c0
            body = String(descr[byte=1:])
    if endian == ">":
        raise Error(
            "npy: `" + descr + "` is big-endian; this reader does not byte-swap"
            " (a silently unswapped read is garbage that looks like data)"
        )
    var item_size: Int
    var is_float: Bool
    if body == "f8":
        item_size = 8
        is_float = True
    elif body == "f4":
        item_size = 4
        is_float = True
    elif body == "i8" or body == "u8":
        item_size = 8
        is_float = False
    elif body == "i4" or body == "u4":
        item_size = 4
        is_float = False
    else:
        raise Error(
            "npy: dtype `" + descr + "` is not one we read — f4/f8/i4/i8 only"
            " (an `O` dtype is a pickle: see io/pickle.mojo)"
        )

    # ── fortran_order ────────────────────────────────────────────────────
    var kf = _find_key(h, String("fortran_order"))
    if kf < 0:
        raise Error("npy: the header has no `fortran_order`")
    if kf + 4 <= len(h) and h[kf] == 84:        # 'T'rue
        raise Error(
            "npy: `fortran_order: True` — the array is column-major, and"
            " reading it as C order transposes it silently"
        )

    # ── shape ────────────────────────────────────────────────────────────
    var ks = _find_key(h, String("shape"))
    if ks < 0:
        raise Error("npy: the header has no `shape`")
    var shape = _read_tuple(h, ks)
    var rows = 1
    var cols = 1
    if len(shape) >= 1:
        rows = shape[0]
        for i in range(1, len(shape)):
            cols *= shape[i]
    var n = rows * cols

    var data_offset = hstart + hlen
    var want = n * item_size
    if data_offset + want > total_size:
        raise Error(
            "npy: the header says " + String(n) + " x " + String(item_size)
            + " bytes of data but the file holds "
            + String(total_size - data_offset) + " after the header"
        )
    return NpyHeader(descr^, item_size, is_float, rows, cols, data_offset, n)


def npy_header_of(path: String) raises -> NpyHeader:
    """The header alone — the shape of a file without reading its data."""
    if not exists(path):
        raise Error("npy: '" + path + "' does not exist")
    var total = file_size(path)
    var want = 256 if total > 256 else total
    return parse_npy_header(read_file_range(path, 0, want), total)


def load_npy_f64(path: String) raises -> List[Float64]:
    """The whole array as float64, row-major, whatever its stored dtype."""
    var all = read_file_bytes(path)
    var hd = parse_npy_header(all, len(all))
    var out = List[Float64](length=hd.n_elems, fill=0.0)
    var src = all.unsafe_ptr().unsafe_offset(hd.data_offset)
    if hd.is_float and hd.item_size == 8:
        var p = src.unsafe_bitcast[Float64]()
        for i in range(hd.n_elems):
            out[i] = p[unsafe_offset=i]
    elif hd.is_float and hd.item_size == 4:
        var p = src.unsafe_bitcast[Float32]()
        for i in range(hd.n_elems):
            out[i] = Float64(p[unsafe_offset=i])
    elif hd.item_size == 8:
        var p = src.unsafe_bitcast[Int64]()
        for i in range(hd.n_elems):
            out[i] = Float64(p[unsafe_offset=i])
    else:
        var p = src.unsafe_bitcast[Int32]()
        for i in range(hd.n_elems):
            out[i] = Float64(p[unsafe_offset=i])
    return out^


struct Npy2D(Movable):
    """A `[rows, cols]` float64 array, row-major."""
    var values: List[Float64]
    var rows: Int
    var cols: Int

    def __init__(out self, var values: List[Float64], rows: Int, cols: Int):
        self.values = values^
        self.rows = rows
        self.cols = cols

    def __init__(out self, *, deinit move: Self):
        self.values = move.values^
        self.rows = move.rows
        self.cols = move.cols


def load_npy_2d_f64(path: String, want_cols: Int) raises -> Npy2D:
    """A `[N, want_cols]` array. Raises on any other width.

    The width is the caller's contract with the file, so it is checked here
    rather than inferred: a `[N, 35]` clip reshaped to 36 columns would slide
    every frame by one element and still look like poses.
    """
    var hd = npy_header_of(path)
    if hd.cols != want_cols:
        raise Error(
            "npy: '" + path + "' is [" + String(hd.rows) + ", "
            + String(hd.cols) + "], not [N, " + String(want_cols) + "]"
        )
    return Npy2D(load_npy_f64(path), hd.rows, hd.cols)
