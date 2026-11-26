"""The mean of several checkpoints (`nn/core/checkpoint_mean.mojo`).

1. Known answer, files built by hand: a parameter with moments, a state, a
   parameter without, two scalars. The output is byte for byte the file this
   test builds from the means it states; the report counts the sections and
   values, gives the largest shift, and names the state and the scalar that
   differ. The scalar written is the LAST input's.
2. The sum is Float64: 2^24, 1 and 1 average to 5592406 exactly; summed in
   float32 the two ones are lost and the mean would be 5592405.5.
3. The mean of a file with itself, three times, is that file byte for byte,
   and the report says nothing moved.
4. The chunk loop: the same means with one and two values per read are the
   same bytes.
5. A file written by `save_params` (Adam's moments populated, a scalar): its
   layout reads, the mean of two such files loads back through `load_params`
   into the same model, each weight is the mean of the two inputs' and not
   either input's (the control).
6. Refusals, each leaving the output as it was and no `.tmp` behind: one
   input, an output that exists (replaced only when asked), an output that is
   an input, sections that differ in size / in number / in the moments flag,
   a v2 file, a truncated file, a mean that is not finite.

Run: pixi run mojo run -I . tests/nn/test_checkpoint_mean.mojo
"""

from std.memory import bitcast
from std.os.path import exists
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from noeira.io.fileio import read_file_bytes, write_file_atomic, remove_file
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.checkpoint import (
    save_params, load_params, CheckpointScalars,
)
from noeira.nn.core.checkpoint_mean import (
    checkpoint_layout, mean_checkpoints,
)
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime NET = Sequential[Linear[D, H], Linear[H, O]]


def _s(mut b: List[UInt8], s: String):
    var src = s.as_bytes()
    for i in range(s.byte_length()):
        b.append(src[i])


def _f32(mut b: List[UInt8], x: Float32):
    var bits = bitcast[DType.uint32](x)
    for i in range(4):
        b.append(UInt8((bits >> UInt32(8 * i)) & 0xFF))


def _f64(mut b: List[UInt8], x: Float64):
    var bits = bitcast[DType.uint64](x)
    for i in range(8):
        b.append(UInt8((bits >> UInt64(8 * i)) & 0xFF))


def _file(
    w: List[Float32], n: List[Float32], bias: Float32, step: Float64
) -> List[UInt8]:
    """`P w 3 1` (3 values, then m, then v: 9 floats), `S n 2`, `P b 1 0`,
    `K step`, `K lr` (always 0.5)."""
    var b = List[UInt8]()
    _s(b, "storage-ckpt v3\nP w 3 1\n")
    for i in range(9):
        _f32(b, w[i])
    _s(b, "S n 2\n")
    for i in range(2):
        _f32(b, n[i])
    _s(b, "P b 1 0\n")
    _f32(b, bias)
    _s(b, "K step\n")
    _f64(b, step)
    _s(b, "K lr\n")
    _f64(b, 0.5)
    return b^


def _write(path: String, b: List[UInt8]) raises:
    write_file_atomic(path, b)


def _same_bytes(a: List[UInt8], b: List[UInt8]) -> Bool:
    if len(a) != len(b) or len(a) == 0:
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _a() -> List[UInt8]:
    return _file([1.0, 2.0, 3.0, 0.5, 0.25, 0.0, 4.0, 4.0, 4.0], [10.0, 20.0], 7.0, 100.0)


def _b() -> List[UInt8]:
    return _file([3.0, 2.0, -3.0, 1.5, 0.25, 1.0, 8.0, 4.0, 0.0], [10.0, 30.0], 8.0, 200.0)


def _mean_ab() -> List[UInt8]:
    return _file([2.0, 2.0, 0.0, 1.0, 0.25, 0.5, 6.0, 4.0, 2.0], [10.0, 25.0], 7.5, 200.0)


def _refused(dst: String, inputs: List[String], overwrite: Bool = False) -> Bool:
    try:
        _ = mean_checkpoints(dst, inputs, overwrite)
    except:
        return True
    return False


def test_known_answer() raises:
    _write("/tmp/ckpt_mean_a.bin", _a())
    _write("/tmp/ckpt_mean_b.bin", _b())
    remove_file("/tmp/ckpt_mean_ab.bin")
    var lay = checkpoint_layout("/tmp/ckpt_mean_a.bin")
    assert_equal(len(lay), 5)
    assert_equal(lay[0].header, "P w 3 1")
    assert_true(lay[0].has_moments and lay[0].size == 3 and lay[0].nbytes == 36)
    assert_equal(lay[0].offset, 24)
    assert_true(lay[1].kind == "S" and lay[1].name == "n" and lay[1].nbytes == 8)
    assert_true(not lay[2].has_moments and lay[2].nbytes == 4)
    assert_true(lay[3].kind == "K" and lay[3].name == "step" and lay[3].nbytes == 8)

    var rep = mean_checkpoints(
        "/tmp/ckpt_mean_ab.bin",
        ["/tmp/ckpt_mean_a.bin", "/tmp/ckpt_mean_b.bin"],
    )
    var got = read_file_bytes("/tmp/ckpt_mean_ab.bin")
    assert_true(_same_bytes(got, _mean_ab()), "the mean's bytes")
    assert_true(not _same_bytes(got, _a()) and not _same_bytes(got, _b()), "the mean is neither input")
    assert_equal(rep.inputs, 2)
    assert_equal(rep.sections, 5)
    assert_equal(rep.values, 12)
    assert_true(rep.max_shift == 5.0, "largest shift: the state, 20 to 25")
    assert_equal(len(rep.states_differing), 1)
    assert_equal(rep.states_differing[0], "n")
    assert_equal(len(rep.scalars_differing), 1)
    assert_equal(rep.scalars_differing[0], "step")
    assert_true(not exists("/tmp/ckpt_mean_ab.bin.tmp"))

    # The scalar is the last input's: the other order keeps 100.
    var rev = mean_checkpoints(
        "/tmp/ckpt_mean_ab.bin",
        ["/tmp/ckpt_mean_b.bin", "/tmp/ckpt_mean_a.bin"],
        overwrite=True,
    )
    var want = _file([2.0, 2.0, 0.0, 1.0, 0.25, 0.5, 6.0, 4.0, 2.0], [10.0, 25.0], 7.5, 100.0)
    assert_true(_same_bytes(read_file_bytes("/tmp/ckpt_mean_ab.bin"), want), "the last input's scalar")
    assert_equal(rev.scalars_differing[0], "step")


def test_sum_is_float64() raises:
    var big = _file([16777216.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0], 0.0, 1.0)
    var one = _file([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0], 0.0, 1.0)
    _write("/tmp/ckpt_mean_big.bin", big)
    _write("/tmp/ckpt_mean_one.bin", one)
    _write("/tmp/ckpt_mean_one2.bin", one)
    var rep = mean_checkpoints(
        "/tmp/ckpt_mean_f64.bin",
        ["/tmp/ckpt_mean_big.bin", "/tmp/ckpt_mean_one.bin", "/tmp/ckpt_mean_one2.bin"],
        overwrite=True,
    )
    var want = _file([5592406.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0], 0.0, 1.0)
    var wrong = _file([5592405.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0], 0.0, 1.0)
    var got = read_file_bytes("/tmp/ckpt_mean_f64.bin")
    assert_true(not _same_bytes(want, wrong), "the two answers are different floats")
    assert_true(_same_bytes(got, want), "a Float64 sum gives 5592406")
    assert_equal(len(rep.scalars_differing), 0)
    assert_equal(len(rep.states_differing), 0)


def test_self_mean_is_identity() raises:
    _write("/tmp/ckpt_mean_a.bin", _a())
    _write("/tmp/ckpt_mean_a2.bin", _a())
    _write("/tmp/ckpt_mean_a3.bin", _a())
    var rep = mean_checkpoints(
        "/tmp/ckpt_mean_self.bin",
        ["/tmp/ckpt_mean_a.bin", "/tmp/ckpt_mean_a2.bin", "/tmp/ckpt_mean_a3.bin"],
        overwrite=True,
    )
    assert_true(_same_bytes(read_file_bytes("/tmp/ckpt_mean_self.bin"), _a()), "self mean")
    assert_true(rep.max_shift == 0.0)
    assert_equal(len(rep.states_differing), 0)
    assert_equal(len(rep.scalars_differing), 0)


def test_chunk_loop() raises:
    _write("/tmp/ckpt_mean_a.bin", _a())
    _write("/tmp/ckpt_mean_b.bin", _b())
    for chunk in [1, 2, 4]:
        var rep = mean_checkpoints(
            "/tmp/ckpt_mean_chunk.bin",
            ["/tmp/ckpt_mean_a.bin", "/tmp/ckpt_mean_b.bin"],
            overwrite=True, chunk_elems=chunk,
        )
        assert_true(
            _same_bytes(read_file_bytes("/tmp/ckpt_mean_chunk.bin"), _mean_ab()),
            "the mean read in small chunks",
        )
        assert_equal(rep.values, 12)
        assert_true(rep.max_shift == 5.0)
        assert_equal(len(rep.states_differing), 1)


struct _Capture(ParamVisitor):
    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            self.vals.append(param.data[i])


def _params(mut net: NET) raises -> List[Scalar[DT]]:
    var c = _Capture()
    net.for_each_param["cpu"](c, None)
    return c.vals.copy()


def _step(mut net: NET, mut opt: Adam, step: Int) raises:
    var x = Tensor.alloc(B * D)
    var go = Tensor.alloc(B * O)
    for i in range(B * D):
        x.data[i] = Scalar[DT](((i + step) % 5) - 2) * 0.3
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.4
    var out = Tensor.alloc(B * O)
    var gi = Tensor.alloc(B * D)
    net.zero_grad["cpu"](None)
    net.forward["cpu", B](TensorRefs[1](x), out, None)
    net.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    opt.step["cpu"](net, None)


def test_real_files_round_trip() raises:
    var net = NET.make["cpu", Deterministic](None)
    var opt = Adam(lr=1e-2)
    opt.adopt["cpu"](net, None)
    for s in range(2):
        _step(net, opt, s)
    var sc = CheckpointScalars()
    sc.set_int("iteration", 2)
    save_params["cpu"](net, "/tmp/ckpt_mean_r1.bin", None, save_moments=True, scalars=sc)
    var p1 = _params(net)
    for s in range(2, 6):
        _step(net, opt, s)
    sc.set_int("iteration", 6)
    save_params["cpu"](net, "/tmp/ckpt_mean_r2.bin", None, save_moments=True, scalars=sc)
    var p2 = _params(net)

    var lay = checkpoint_layout("/tmp/ckpt_mean_r1.bin")
    var n_p = 0
    for k in range(len(lay)):
        if lay[k].kind == "P":
            assert_true(lay[k].has_moments, "the moments were saved")
            n_p += lay[k].size
    assert_equal(n_p, len(p1))
    assert_true(lay[len(lay) - 1].kind == "K" and lay[len(lay) - 1].name == "iteration")

    var rep = mean_checkpoints(
        "/tmp/ckpt_mean_r.bin",
        ["/tmp/ckpt_mean_r1.bin", "/tmp/ckpt_mean_r2.bin"],
        overwrite=True,
    )
    assert_equal(rep.values, 3 * len(p1))
    assert_true(rep.max_shift > 0.0)
    var back = NET.make["cpu", Deterministic](None)
    var k = load_params["cpu"](back, "/tmp/ckpt_mean_r.bin", None)
    assert_equal(k.get_int("iteration", 0), 6)
    var pm = _params(back)
    assert_equal(len(pm), len(p1))
    var moved = 0
    for i in range(len(pm)):
        var want = Scalar[DT]((Float64(p1[i]) + Float64(p2[i])) / 2.0)
        assert_true(pm[i] == want, "a weight of the mean")
        if pm[i] != p1[i] and pm[i] != p2[i]:
            moved += 1
    print("  real files:", len(pm), "weights,", moved, "differ from both inputs; max shift", rep.max_shift)
    assert_true(moved > len(pm) // 2, "the mean is neither input — else vacuous")

    # A copy saved without its moments is not the same file layout.
    save_params["cpu"](net, "/tmp/ckpt_mean_r3.bin", None, save_moments=False, scalars=sc)
    assert_true(
        _refused("/tmp/ckpt_mean_r.bin", ["/tmp/ckpt_mean_r1.bin", "/tmp/ckpt_mean_r3.bin"], True),
        "with and without moments do not average",
    )
    _ = load_params["cpu"](back, "/tmp/ckpt_mean_r.bin", None)
    assert_true(_params(back)[0] == pm[0], "a refusal leaves the output as it was")


def test_refusals() raises:
    var a = String("/tmp/ckpt_mean_a.bin")
    var b = String("/tmp/ckpt_mean_b.bin")
    var out = String("/tmp/ckpt_mean_ref.bin")
    _write(a, _a())
    _write(b, _b())
    remove_file(out)

    assert_true(_refused(out, [a]), "one input")
    assert_true(not exists(out))

    # An output that exists stays, unless asked.
    _write(out, _a())
    assert_true(_refused(out, [a, b]), "an existing output")
    assert_true(_same_bytes(read_file_bytes(out), _a()), "and it is untouched")
    assert_true(not _refused(out, [a, b], True), "replaced when asked")
    assert_true(_same_bytes(read_file_bytes(out), _mean_ab()))

    # The output is never an input, asked or not.
    assert_true(_refused(a, [a, b], True), "the output is an input")
    assert_true(_same_bytes(read_file_bytes(a), _a()))

    # Sections that differ: a size, a count, a kind of file, a short file.
    var sized = List[UInt8]()
    _s(sized, "storage-ckpt v3\nP w 4 1\n")
    for _ in range(12):
        _f32(sized, 1.0)
    _write("/tmp/ckpt_mean_sized.bin", sized)
    assert_true(_refused(out, [a, "/tmp/ckpt_mean_sized.bin"], True), "a size differs")

    var fewer = List[UInt8]()
    _s(fewer, "storage-ckpt v3\nP w 3 1\n")
    for _ in range(9):
        _f32(fewer, 1.0)
    _write("/tmp/ckpt_mean_fewer.bin", fewer)
    assert_true(_refused(out, [a, "/tmp/ckpt_mean_fewer.bin"], True), "fewer sections")
    assert_true(_refused(out, ["/tmp/ckpt_mean_fewer.bin", a], True), "more sections")

    var v2 = List[UInt8]()
    _s(v2, "storage-ckpt v2\nP w 1 0\n1.0\n")
    _write("/tmp/ckpt_mean_v2.bin", v2)
    assert_true(_refused(out, [a, "/tmp/ckpt_mean_v2.bin"], True), "a v2 file")

    var cut = _a()
    cut.resize(len(cut) - 3, 0)
    _write("/tmp/ckpt_mean_cut.bin", cut)
    assert_true(_refused(out, [a, "/tmp/ckpt_mean_cut.bin"], True), "a truncated file")

    # A mean that is not finite: +inf and -inf in the same place.
    var inf = Float32(1.0e30) * Float32(1.0e30)
    _write("/tmp/ckpt_mean_pinf.bin", _file([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, inf], 0.0, 1.0))
    _write("/tmp/ckpt_mean_ninf.bin", _file([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -inf], 0.0, 1.0))
    assert_true(
        _refused(out, ["/tmp/ckpt_mean_pinf.bin", "/tmp/ckpt_mean_ninf.bin"], True),
        "a mean that is not finite",
    )

    # Every refusal above left the output of the one accepted call, and no tmp.
    assert_true(_same_bytes(read_file_bytes(out), _mean_ab()), "refusals leave the output")
    assert_true(not exists(out + ".tmp"), "and no .tmp")


def main() raises:
    test_known_answer()
    test_sum_is_float64()
    test_self_mean_is_identity()
    test_chunk_loop()
    test_real_files_round_trip()
    test_refusals()
    print("CHECKPOINT MEAN OK")
