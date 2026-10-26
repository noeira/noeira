"""A frozen param in a v3 checkpoint: same names, same places, no moments.

    pixi run mojo run -I . tests/nn/peft/test_frozen_checkpoint.mojo

Requirement R4 of the frozen-weight design: freezing a layer must not change
how a model is saved or loaded. A frozen `Param` keeps its `P` section at its
place in the walk (v3 reads sections POSITIONALLY, so a frozen weight moved to
the `S` role would break every existing checkpoint) and is written with
`has_moments = 0`, because it has none.

The model is the SmolVLA expert at toy width, the unit `save_trainables`
actually writes. Two of its linears are frozen, every other param is given
moments (as after an Adam step), then:

  [1] the file: the frozen params' sections say `has_moments 0`, every other
      section says 1, and the section names are those of the unfrozen walk
  [2] load into a TRAINABLE model: every value bit-identical; moments back
      wherever the file had them; the formerly frozen params trainable here
  [3] load into a model with the SAME layers frozen: values bit-identical,
      the frozen params still frozen, no moments allocated for them
  [4] an OLD checkpoint (everything trainable, moments everywhere) into the
      frozen model: it loads, values bit-identical, and the reader reads past
      the frozen params' moments instead of allocating them again
  [5] the same with a v2 (text) checkpoint: the other reader
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.core.bytes import string_from_bytes
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor, ParamVisitorRT
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.checkpoint import (
    save_params, load_params, LegacyV2CheckpointWriter,
)
from noeira.nn.core.param import walk_params
from noeira.io.fileio import read_file_bytes
from noeira.deep_agents.smolvla.expert import SmolVLAExpert

comptime Expert = SmolVLAExpert[2, 8, 12, 8, 4, 2]


struct Probe(Movable, ParamVisitor, ParamVisitorRT):
    """mode 0: give every trainable param distinct values and moments.
    mode 1: record values, moment sizes and frozen flags, in walk order."""

    var mode: Int
    var k: Int
    var vals: List[Scalar[DT]]
    var m_sizes: List[Int]
    var frozen: List[Bool]
    var names: List[String]

    def __init__(out self, mode: Int):
        self.mode = mode
        self.k = 0
        self.vals = List[Scalar[DT]]()
        self.m_sizes = List[Int]()
        self.frozen = List[Bool]()
        self.names = List[String]()

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if self.mode == 0:
            for i in range(n):
                param.data[i] = Scalar[DT](0.001) * Scalar[DT](
                    (self.k * 31 + i * 7) % 97 - 48
                )
            if grad.n > 0:
                m.ensure(n)
                v.ensure(n)
                for i in range(n):
                    m.data[i] = Scalar[DT](0.5) * param.data[i]
                    v.data[i] = Scalar[DT](0.25) * param.data[i] * param.data[i]
            self.k += 1
            return
        for i in range(n):
            self.vals.append(param.data[i])
        self.m_sizes.append(m.n)
        self.frozen.append(grad.n == 0)
        self.names.append(name)

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


def _freeze_two(mut e: Expert):
    e.self_layers[0].q.freeze()
    e.cross_layers[0].mlp.down.freeze()


def _record(mut e: Expert) raises -> Probe:
    var p = Probe(1)
    e.for_each_param["cpu"](p, None)
    return p^


def _section_flags(path: String) raises -> List[String]:
    """`<name> <has_moments>` for every P section header of a v3 file."""
    var b = read_file_bytes(path)
    var out = List[String]()
    var i = 0
    var n = len(b)
    # Skip the "storage-ckpt v3" line, then hop header -> payload -> header.
    while i < n and b[i] != UInt8(10):
        i += 1
    i += 1
    while i < n:
        var line = List[UInt8]()
        while i < n and b[i] != UInt8(10):
            line.append(b[i])
            i += 1
        i += 1
        var s = string_from_bytes(line)
        var toks = s.split(" ")
        if String(toks[0]) != "P":
            break  # S or K sections follow; this expert has none
        var size = atol(String(toks[2]))
        var has_m = String(toks[3]) == "1"
        out.append(String(toks[1]) + " " + String(toks[3]))
        i += size * 4 * (3 if has_m else 1)
    return out^


def main() raises:
    print("frozen params through a v3 checkpoint")
    var path = String("/tmp/noeira_test_frozen_ckpt.bin")
    var path_old = String("/tmp/noeira_test_frozen_ckpt_old.bin")

    # The source model: values + moments everywhere, then two linears frozen.
    var src = Expert.make["cpu", Deterministic]()
    var seed = Probe(0)
    src.for_each_param["cpu"](seed, None)
    save_params["cpu"](src, path_old)  # the OLD file: nothing frozen yet
    _freeze_two(src)
    save_params["cpu"](src, path)
    var ref_ = _record(src)
    var n_frozen = 0
    for f in ref_.frozen:
        if f:
            n_frozen += 1

    # [1] the file
    var flags = _section_flags(path)
    var bad_flag = 0
    for j in range(len(flags)):
        var want = String(ref_.names[j]) + (" 0" if ref_.frozen[j] else " 1")
        if flags[j] != want:
            bad_flag += 1
    print("  [1] sections", len(flags), " params walked", len(ref_.names),
          " frozen", n_frozen, " header mismatches", bad_flag)
    assert_true(n_frozen == 4, "expected 2 linears x (weight, bias) frozen")
    assert_true(len(flags) == len(ref_.names), "a frozen param lost its section")
    assert_true(bad_flag == 0, "a section's name or has_moments flag is wrong")

    # [2] into a trainable model
    var dst = Expert.make["cpu", Deterministic]()
    _ = load_params["cpu"](dst, path)
    var got = _record(dst)
    var bad_v = 0
    for i in range(len(ref_.vals)):
        if got.vals[i] != ref_.vals[i]:
            bad_v += 1
    var bad_m = 0
    var trainable_now = 0
    for j in range(len(ref_.names)):
        var want_m = 0 if ref_.frozen[j] else ref_.m_sizes[j]
        if got.m_sizes[j] != want_m:
            bad_m += 1
        if ref_.frozen[j] and not got.frozen[j]:
            trainable_now += 1
    print("  [2] trainable model: value mismatches", bad_v, " moment-size"
          " mismatches", bad_m, " formerly frozen now trainable", trainable_now)
    assert_true(bad_v == 0, "values changed through the checkpoint")
    assert_true(bad_m == 0, "moments did not come back where the file had them")
    assert_true(trainable_now == n_frozen, "a frozen param stayed frozen")

    # [3] into a model with the same layers frozen
    var dstf = Expert.make["cpu", Deterministic]()
    _freeze_two(dstf)
    _ = load_params["cpu"](dstf, path)
    var gotf = _record(dstf)
    var bad_vf = 0
    for i in range(len(ref_.vals)):
        if gotf.vals[i] != ref_.vals[i]:
            bad_vf += 1
    var leaked = 0
    var unfrozen = 0
    for j in range(len(ref_.names)):
        if ref_.frozen[j]:
            if gotf.m_sizes[j] != 0:
                leaked += 1
            if not gotf.frozen[j]:
                unfrozen += 1
    print("  [3] frozen model: value mismatches", bad_vf, " frozen params"
          " un-frozen", unfrozen, " with moments", leaked)
    assert_true(bad_vf == 0, "values changed through the checkpoint")
    assert_true(unfrozen == 0, "loading un-froze a frozen param")
    assert_true(leaked == 0, "loading gave a frozen param moments")

    # [4] an old checkpoint (moments everywhere) into the frozen model
    var dsto = Expert.make["cpu", Deterministic]()
    _freeze_two(dsto)
    _ = load_params["cpu"](dsto, path_old)
    var goto = _record(dsto)
    var bad_vo = 0
    for i in range(len(ref_.vals)):
        if goto.vals[i] != ref_.vals[i]:
            bad_vo += 1
    var alloc_m = 0
    for j in range(len(ref_.names)):
        if ref_.frozen[j] and goto.m_sizes[j] > 0:
            alloc_m += 1
    print("  [4] old checkpoint into the frozen model: value mismatches",
          bad_vo, " frozen params given moments by the reader", alloc_m)
    assert_true(bad_vo == 0, "an old checkpoint loads different values")
    assert_true(alloc_m == 0, "the v3 reader gave a frozen param moments")

    # [5] the v2 text reader, same old-checkpoint case
    var path_v2 = String("/tmp/noeira_test_frozen_ckpt_v2.txt")
    var old = Expert.make["cpu", Deterministic]()
    var seed2 = Probe(0)
    old.for_each_param["cpu"](seed2, None)
    var w2 = LegacyV2CheckpointWriter(True)
    walk_params["cpu"](old, w2, None)
    with open(path_v2, "w") as fh:
        fh.write(w2.content)
    var dst2 = Expert.make["cpu", Deterministic]()
    _freeze_two(dst2)
    _ = load_params["cpu"](dst2, path_v2)
    var got2 = _record(dst2)
    var bad_v2 = 0
    for i in range(len(ref_.vals)):
        if got2.vals[i] != ref_.vals[i]:
            bad_v2 += 1
    var alloc_v2 = 0
    var trained_m = 0
    for j in range(len(ref_.names)):
        if ref_.frozen[j] and got2.m_sizes[j] > 0:
            alloc_v2 += 1
        if not ref_.frozen[j] and got2.m_sizes[j] > 0:
            trained_m += 1
    print("  [5] v2 checkpoint into the frozen model: value mismatches", bad_v2,
          " frozen params given moments", alloc_v2, " trainable ones with moments",
          trained_m, "of", len(ref_.names) - n_frozen)
    assert_true(bad_v2 == 0, "a v2 checkpoint loads different values")
    assert_true(alloc_v2 == 0, "the v2 reader gave a frozen param moments")
    assert_true(trained_m == len(ref_.names) - n_frozen,
                "the v2 reader lost a trainable param's moments")

    print("PASSED")
