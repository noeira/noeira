"""The mean of several copies of one run, as a checkpoint — no model needed.

    var rep = mean_checkpoints("mean.ckpt", ["a.ckpt", "b.ckpt", "c.ckpt"])

Every `P` and `S` section of the output is the element-wise mean of the
inputs' (summed in Float64 in the order given, written as `Scalar[DT]`); every
`K` scalar is the LAST input's. The inputs must be v3 files with the same
sections in the same order: the same model, saved at different moments.

What it is for. A run keeps a few copies near its end, and which one is best
moves from copy to copy. On the G1 object-carrying runs (ten runs, four kept
copies each, 5,000 iterations apart) the mean of the four was at or above the
run's median copy in 10, 10 and 8 runs of 10 on three held-out measures, and
at or above the run's best copy in 7 runs of 10 on each. That is the only case
measured: copies of ONE run, close in training. Nothing here says the mean of
two different runs is a policy.

What the output is not. It is a copy to evaluate or deploy, not a point to
resume training from: Adam's moments, when the inputs hold them, are averaged
like the weights (so the file keeps the inputs' layout and loads into the same
model), and the counters are the last input's. `S` sections are averaged too,
and the report names those that differed between the inputs: a state that is
a count rather than a statistic would not survive a mean.

It works on the file's sections alone (`checkpoint_layout`), so it needs no
model type and holds `chunk_elems` values per input at a time, whatever the
size of the files.
"""

from std.math import isfinite
from std.memory import unsafe_memcpy
from std.os.path import exists
from std.sys.info import size_of

from noeira.core.bytes import string_from_bytes
from noeira.io.fileio import (
    file_size, read_file_range, remove_file, rename_over,
)
from noeira.nn.constants import DT

comptime _HEADER = "storage-ckpt v3\n"

comptime _MEAN_CHUNK_ELEMS = 1 << 22
"""Values read per input at a time (16 MB of fp32)."""


@fieldwise_init
struct CheckpointSection(Copyable, Movable):
    """One section of a v3 file, as the file states it."""
    var kind: String
    """"P" (a parameter), "S" (a state) or "K" (a Float64 scalar)."""
    var name: String
    var header: String
    """The section's header line, without its newline."""
    var size: Int
    """Values of the tensor; 0 for a `K` section."""
    var has_moments: Bool
    """A `P` section followed by Adam's m and v (two more tensors)."""
    var offset: Int
    """The payload's first byte in the file."""
    var nbytes: Int
    """The payload's length: `size` values, three times that with moments,
    8 bytes for a `K` section."""


def _line_at(path: String, offset: Int, total: Int) raises -> String:
    """The line starting at `offset`, without its newline."""
    var window = 256
    while True:
        var take = min(window, total - offset)
        var b = read_file_range(path, offset, take)
        for i in range(len(b)):
            if b[i] == UInt8(10):
                b.resize(i, 0)
                return string_from_bytes(b)
        if take == total - offset:
            raise Error(
                "checkpoint: '" + path + "' ends inside a section header at"
                " byte " + String(offset)
            )
        window *= 2


def checkpoint_layout(path: String) raises -> List[CheckpointSection]:
    """The sections of a v3 checkpoint in file order, read from the headers
    alone (no payload is loaded). Raises on a file that is not v3, on a header
    it cannot read and on a payload that runs past the end of the file."""
    comptime SB = size_of[Scalar[DT]]()
    var total = file_size(path)
    var tag = String(_HEADER)
    var tb = tag.as_bytes()
    var ok = total >= len(tb)
    if ok:
        var head = read_file_range(path, 0, len(tb))
        for i in range(len(tb)):
            if head[i] != tb[i]:
                ok = False
    if not ok:
        raise Error(
            "checkpoint: '" + path + "' is not a `storage-ckpt v3` file"
        )
    var out = List[CheckpointSection]()
    var cur = len(tb)
    while cur < total:
        var line = _line_at(path, cur, total)
        var toks = line.split(" ")
        var kind = String(toks[0])
        var size = 0
        var has_m = False
        var nbytes: Int
        if kind == "K" and len(toks) == 2:
            nbytes = 8
        elif kind == "S" and len(toks) == 3:
            size = Int(atol(toks[2]))
            nbytes = size * SB
        elif kind == "P" and len(toks) == 4 and (
            String(toks[3]) == "0" or String(toks[3]) == "1"
        ):
            size = Int(atol(toks[2]))
            has_m = String(toks[3]) == "1"
            nbytes = size * SB * (3 if has_m else 1)
        else:
            raise Error(
                "checkpoint: '" + path + "' has an unreadable section header"
                " `" + line + "` at byte " + String(cur)
            )
        var offset = cur + line.byte_length() + 1
        if size < 0 or offset + nbytes > total:
            raise Error(
                "checkpoint: '" + path + "' is truncated: section `" + line
                + "` needs " + String(nbytes) + " bytes from byte "
                + String(offset) + ", the file has " + String(total)
            )
        out.append(
            CheckpointSection(
                kind, String(toks[1]), line, size, has_m, offset, nbytes
            )
        )
        cur = offset + nbytes
    return out^


struct CheckpointMeanReport(Copyable, Movable):
    """What `mean_checkpoints` wrote."""
    var inputs: Int
    var sections: Int
    var values: Int
    """Values averaged, moments included."""
    var max_shift: Float64
    """The largest |mean - first input| over those values. 0 means the inputs
    held the same tensors: the output is then the first input's, and nothing
    was averaged in effect."""
    var states_differing: List[String]
    """`S` sections whose mean differs from the first input's values."""
    var scalars_differing: List[String]
    """`K` scalars that are not the same in every input (the last input's is
    the one written)."""

    def __init__(out self, inputs: Int):
        self.inputs = inputs
        self.sections = 0
        self.values = 0
        self.max_shift = 0.0
        self.states_differing = List[String]()
        self.scalars_differing = List[String]()


def _put(mut f: FileHandle, b: List[UInt8]) raises:
    f.write_bytes(Span(b))


def _put_str(mut f: FileHandle, s: String) raises:
    var b = List[UInt8]()
    var src = s.as_bytes()
    for i in range(s.byte_length()):
        b.append(src[i])
    _put(f, b)


def _write_mean(
    mut f: FileHandle,
    inputs: List[String],
    first: List[CheckpointSection],
    chunk_elems: Int,
    mut rep: CheckpointMeanReport,
) raises:
    comptime SB = size_of[Scalar[DT]]()
    var n_in = len(inputs)
    _put_str(f, String(_HEADER))
    for k in range(len(first)):
        var sec = first[k].copy()
        _put_str(f, sec.header + "\n")
        rep.sections += 1
        if sec.kind == "K":
            var ref_bytes = read_file_range(inputs[0], sec.offset, 8)
            var last = ref_bytes.copy()
            var differ = False
            for j in range(1, n_in):
                last = read_file_range(inputs[j], sec.offset, 8)
                for i in range(8):
                    if last[i] != ref_bytes[i]:
                        differ = True
            if differ:
                rep.scalars_differing.append(sec.name)
            _put(f, last)
            continue
        var n_total = sec.nbytes // SB
        var moved = False
        var off = 0
        while off < n_total:
            var take = min(chunk_elems, n_total - off)
            var acc = List[Float64](length=take, fill=0.0)
            var vals = List[Scalar[DT]](length=take, fill=0.0)
            var lead = List[Scalar[DT]](length=take, fill=0.0)
            for j in range(n_in):
                var b = read_file_range(
                    inputs[j], sec.offset + off * SB, take * SB
                )
                unsafe_memcpy(
                    dest=vals.unsafe_ptr().unsafe_bitcast[UInt8](),
                    src=b.unsafe_ptr(),
                    count=take * SB,
                )
                for i in range(take):
                    acc[i] += Float64(vals[i])
                if j == 0:
                    for i in range(take):
                        lead[i] = vals[i]
            for i in range(take):
                var m = Scalar[DT](acc[i] / Float64(n_in))
                if not isfinite(m):
                    raise Error(
                        "checkpoint mean: section `" + sec.header + "` has a"
                        " value that is not finite at index " + String(off + i)
                    )
                var d = abs(Float64(m) - Float64(lead[i]))
                if d > 0.0:
                    moved = True
                    if d > rep.max_shift:
                        rep.max_shift = d
                vals[i] = m
            var ob = List[UInt8](length=take * SB, fill=0)
            unsafe_memcpy(
                dest=ob.unsafe_ptr(),
                src=vals.unsafe_ptr().unsafe_bitcast[UInt8](),
                count=take * SB,
            )
            _put(f, ob)
            off += take
        rep.values += n_total
        if moved and sec.kind == "S":
            rep.states_differing.append(sec.name)


def mean_checkpoints(
    out_path: String,
    inputs: List[String],
    overwrite: Bool = False,
    chunk_elems: Int = _MEAN_CHUNK_ELEMS,
) raises -> CheckpointMeanReport:
    """Write to `out_path` the mean of the v3 checkpoints `inputs` (two or
    more, same sections in the same order). Atomic: the file is built at
    `out_path + ".tmp"` and renamed at the end, so a refusal or a failure
    leaves `out_path` as it was.

    Refuses an `out_path` that exists unless `overwrite`, and one that is an
    input in any case: the argument most easily misplaced is the output, and
    the file it would replace is a copy of a run."""
    if len(inputs) < 2:
        raise Error("checkpoint mean: needs two inputs or more")
    if chunk_elems < 1:
        raise Error("checkpoint mean: chunk_elems must be positive")
    for j in range(len(inputs)):
        if inputs[j] == out_path:
            raise Error(
                "checkpoint mean: the output '" + out_path + "' is one of"
                " the inputs"
            )
    if exists(out_path) and not overwrite:
        raise Error(
            "checkpoint mean: '" + out_path + "' exists; it is not replaced"
            " unless asked (`overwrite`; `--force` on the command line)"
        )
    var first = checkpoint_layout(inputs[0])
    for j in range(1, len(inputs)):
        var other = checkpoint_layout(inputs[j])
        for k in range(min(len(first), len(other))):
            if first[k].header != other[k].header:
                raise Error(
                    "checkpoint mean: section " + String(k) + " is `"
                    + first[k].header + "` in '" + inputs[0] + "' and `"
                    + other[k].header + "` in '" + inputs[j]
                    + "': not copies of one model"
                )
        if len(first) != len(other):
            raise Error(
                "checkpoint mean: '" + inputs[0] + "' has "
                + String(len(first)) + " sections and '" + inputs[j]
                + "' has " + String(len(other)) + ": not copies of one model"
            )
    var rep = CheckpointMeanReport(len(inputs))
    var tmp = out_path + ".tmp"
    var f = open(tmp, "w")
    try:
        _write_mean(f, inputs, first, chunk_elems, rep)
    except e:
        f.close()
        remove_file(tmp)
        raise e^
    f.close()
    rename_over(tmp, out_path)
    return rep^
