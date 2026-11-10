# +--------------------------------------------------------------------------+ #
# | Kokoro-82M — the weights, by name
# +--------------------------------------------------------------------------+ #
"""`KokoroWeights` — every tensor of `kokoro-v1_0.safetensors`, looked up by the
checkpoint's own name.

    var w = KokoroWeights.load(kokoro_dir() + "/kokoro-v1_0.safetensors")
    var i = w.at("predictor.duration_proj.linear_layer.weight")   # raises if absent
    ... w.tensors[i].data, w.dim(i, 0) ...

The file is `tools/kokoro/convert_kokoro_checkpoint.py`'s flat copy of the
published pickle (548 fp32 tensors, keys `<group>.<name>`).

## Why a table and not a tree of `nn` leaves (as Parakeet's weights are)

Kokoro is 81 M parameters in 548 tensors of ~40 different shapes — an ALBERT,
six bidirectional LSTMs, AdaIN residual blocks of five widths, a vocoder with
transposed convolutions — and its forward is not GEMM-shaped: most of it is
1-D convolutions over a runtime number of frames. A typed leaf per tensor
would be a dozen generic structs and their walks for a model whose value is
running without onnxruntime, not being fine-tuned. So the tensors keep the
checkpoint's names AND LAYOUTS (a `nn.Linear` weight stays `[out, in]`; the
forward multiplies by its transpose), and the forward asks for them by name.
If a part of it is ever trained, that part becomes leaves then.

## Weight normalisation is folded at load

89 convolutions are stored as `weight_g` / `weight_v` (torch's `weight_norm`,
`dim = 0`): the weight is `g[i] * v[i] / ||v[i]||`, the norm over everything
but the first axis. Each pair becomes one tensor named `<prefix>.weight`, and
the pair is dropped: 548 tensors in the file, 459 in the table.

## Every tensor accounted for

`at` marks a tensor as used. After a full forward, `unused()` lists what was
never asked for — which must be exactly the tensors the model does not read at
inference (ALBERT's pooler). A name the forward gets wrong raises at `at`; a
tensor the forward forgets shows up in `unused()`.
"""

from std.math import sqrt

from noeira.io.safetensors import SafeTensors
from noeira.nn.core.tensor import Tensor


comptime KK_FILE_TENSORS: Int = 548
comptime KK_WEIGHT_NORMED: Int = 89
comptime KK_TABLE_TENSORS: Int = KK_FILE_TENSORS - KK_WEIGHT_NORMED


struct KokoroWeights(Movable):
    var names: List[String]
    var tensors: List[Tensor]
    var shape_data: List[Int]
    var shape_start: List[Int]
    var shape_rank: List[Int]
    var used: List[Bool]
    var index: Dict[String, Int]
    var folded: Int
    """Weight-normalised pairs folded at load."""

    def __init__(out self):
        self.names = List[String]()
        self.tensors = List[Tensor]()
        self.shape_data = List[Int]()
        self.shape_start = List[Int]()
        self.shape_rank = List[Int]()
        self.used = List[Bool]()
        self.index = Dict[String, Int]()
        self.folded = 0

    def _add(mut self, var name: String, ref shape: List[Int], ref values: List[Float32]) raises:
        var n = 1
        for i in range(len(shape)):
            n *= shape[i]
        if n != len(values):
            raise Error("kokoro: '" + name + "' holds " + String(len(values)) + " values, its shape says " + String(n))
        if name in self.index:
            raise Error("kokoro: '" + name + "' appears twice")
        var t = Tensor.alloc(n)
        for i in range(n):
            t.data[i] = values[i]
        self.index[name] = len(self.names)
        self.shape_start.append(len(self.shape_data))
        self.shape_rank.append(len(shape))
        for i in range(len(shape)):
            self.shape_data.append(shape[i])
        self.names.append(name^)
        self.tensors.append(t^)
        self.used.append(False)

    @staticmethod
    def load(path: String) raises -> Self:
        """Read the converted checkpoint, folding every weight-normalised pair.
        Raises unless the file is the one this port was written against (548
        tensors, 89 pairs)."""
        var st = SafeTensors(path)
        if len(st.names) != KK_FILE_TENSORS:
            raise Error(
                "kokoro: " + path + " holds " + String(len(st.names))
                + " tensors, expected " + String(KK_FILE_TENSORS)
            )
        var w = Self()
        for i in range(len(st.names)):
            var name = String(st.names[i])
            if name.endswith(".weight_g"):
                continue  # consumed with its weight_v
            if name.endswith(".weight_v"):
                var prefix = String(name[byte=0 : name.byte_length() - 9])  # drop ".weight_v"
                var g_name = prefix + ".weight_g"
                if not st.has(g_name):
                    raise Error("kokoro: '" + name + "' has no weight_g")
                var v = st.read_f32(name)
                var g = st.read_f32(g_name)
                var shape = st.shape(name)
                var rows = shape[0]
                if len(g) != rows:
                    raise Error("kokoro: '" + g_name + "' is not one gain per row of '" + name + "'")
                var per = len(v) // rows
                for r in range(rows):
                    var sq = 0.0
                    for k in range(per):
                        var x = Float64(v[r * per + k])
                        sq += x * x
                    var scale = Float32(Float64(g[r]) / sqrt(sq))
                    for k in range(per):
                        v[r * per + k] = v[r * per + k] * scale
                w._add(prefix + ".weight", shape, v)
                w.folded += 1
            else:
                var shape = st.shape(name)
                var values = st.read_f32(name)
                w._add(name^, shape, values)
        if w.folded != KK_WEIGHT_NORMED or len(w.names) != KK_TABLE_TENSORS:
            raise Error(
                "kokoro: folded " + String(w.folded) + " weight-normalised pairs into "
                + String(len(w.names)) + " tensors, expected " + String(KK_WEIGHT_NORMED)
                + " and " + String(KK_TABLE_TENSORS)
            )
        return w^

    def at(mut self, name: String) raises -> Int:
        """Index of tensor `name`, marked as used. Raises when there is none."""
        if name not in self.index:
            raise Error("kokoro: no tensor named '" + name + "' in the checkpoint")
        var i = self.index[name]
        self.used[i] = True
        return i

    def has(self, name: String) -> Bool:
        return name in self.index

    def rank(self, i: Int) -> Int:
        return self.shape_rank[i]

    def dim(self, i: Int, axis: Int) -> Int:
        return self.shape_data[self.shape_start[i] + axis]

    def numel(self, i: Int) -> Int:
        return self.tensors[i].n

    def shape_str(self, i: Int) -> String:
        var s = String("[")
        for a in range(self.shape_rank[i]):
            if a > 0:
                s += ", "
            s += String(self.dim(i, a))
        return s + "]"

    def unused(self) -> List[String]:
        """Tensors no `at` has asked for yet."""
        var out = List[String]()
        for i in range(len(self.names)):
            if not self.used[i]:
                out.append(self.names[i].copy())
        return out^

    def parameters(self) -> Int:
        var n = 0
        for i in range(len(self.tensors)):
            n += self.tensors[i].n
        return n
