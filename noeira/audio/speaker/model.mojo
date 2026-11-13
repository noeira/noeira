# +--------------------------------------------------------------------------+ #
# | WeSpeaker ResNet34 — a voice as 256 numbers
# +--------------------------------------------------------------------------+ #
"""The speaker-embedding network (docs/ROBOT_BRAIN_PLAN.md, step B4):
WeSpeaker's ResNet34 trained on VoxCeleb (CC BY 4.0), 6.6 M weights, from
`tools/speaker/extract_wespeaker.py`'s file.

    var net = SpeakerNet.load(dir + "/resnet34.safetensors")
    var feats = kaldi_fbank(pcm_int16_range(wav.samples))     # noeira.audio.speaker.fbank
    subtract_mean(feats)
    var emb = net.embed(feats)                                 # 256 values
    cosine(emb, other)                                         # the same voice: high

THE NETWORK, on the features as one picture `[1, 80, T]` (frequency down,
time across):

    stem     conv 3x3, 32 channels, ReLU
    stage 1  3 blocks, 32 channels              -> [32, 80, T]
    stage 2  4 blocks, 64 channels, stride 2    -> [64, 40, T/2]
    stage 3  6 blocks, 128 channels, stride 2   -> [128, 20, T/4]
    stage 4  3 blocks, 256 channels, stride 2   -> [256, 10, T/8]
    pool     per (channel, frequency) row: the mean and the standard deviation
             over time (n - 1 in the divisor, + 1e-7 under the root) -> 5120
    seg      linear 5120 -> 256, minus the file's `mean_vec`

A block is conv 3x3 (the stage's stride in its first block), ReLU, conv 3x3,
plus its input (through a 1x1 conv of the same stride where the shape
changes), ReLU. The batch norms are folded into the convolutions in the
published file, so every convolution has a bias and there is no norm here.

THIS FILE IS THE CPU FORWARD, the one the reference gate runs
(`tests/audio/test_speaker_reference.mojo`): plain loops, about 2.4 G
multiply-adds a second of speech, one task per output channel.
"""

from std.math import sqrt
from max.algorithm import parallelize

from noeira.io.safetensors import SafeTensors


comptime SPK_MELS: Int = 80
comptime SPK_EMB: Int = 256
comptime SPK_STAGES: Int = 4


struct Plane(Movable):
    """Activations `[c, h, w]`, row-major."""

    var c: Int
    var h: Int
    var w: Int
    var v: List[Float32]

    def __init__(out self, c: Int, h: Int, w: Int):
        self.c = c
        self.h = h
        self.w = w
        self.v = List[Float32](length=c * h * w, fill=0.0)


struct ConvW(Movable):
    var cout: Int
    var cin: Int
    var k: Int
    var w: List[Float32]
    var b: List[Float32]

    def __init__(out self, cout: Int, cin: Int, k: Int, var w: List[Float32], var b: List[Float32]):
        self.cout = cout
        self.cin = cin
        self.k = k
        self.w = w^
        self.b = b^


def conv(ref x: Plane, ref cw: ConvW, stride: Int, relu: Bool) -> Plane:
    """`k x k` convolution (k = 3 with a border of 1, or k = 1 with none),
    its bias, and a ReLU when asked. One task per output channel
    (`parallelize`) when there is enough work to pay for it."""
    var pad = 1 if cw.k == 3 else 0
    var ho = (x.h + 2 * pad - cw.k) // stride + 1
    var wo = (x.w + 2 * pad - cw.k) // stride + 1
    var y = Plane(cw.cout, ho, wo)
    var xp = x.v.unsafe_ptr()
    var yp = y.v.unsafe_ptr()
    var wp = cw.w.unsafe_ptr()
    var bp = cw.b.unsafe_ptr()
    var cin = cw.cin
    var k = cw.k
    var xh = x.h
    var xw = x.w

    def _channel(co: Int) {imm}:
        var obase = co * ho * wo
        var bias = bp[unsafe_offset=co]
        for i in range(ho * wo):
            yp[unsafe_offset=obase + i] = bias
        for ci in range(cin):
            var ibase = ci * xh * xw
            for ky in range(k):
                for kx in range(k):
                    var wv = wp[unsafe_offset=((co * cin + ci) * k + ky) * k + kx]
                    # the output columns whose input column is inside the picture
                    var ox0 = 0
                    while ox0 * stride + kx - pad < 0:
                        ox0 += 1
                    var ox1 = wo
                    while ox1 > ox0 and (ox1 - 1) * stride + kx - pad >= xw:
                        ox1 -= 1
                    for oy in range(ho):
                        var iy = oy * stride + ky - pad
                        if iy < 0 or iy >= xh:
                            continue
                        var o = obase + oy * wo
                        var i = ibase + iy * xw + kx - pad
                        if stride == 1:
                            for ox in range(ox0, ox1):
                                yp[unsafe_offset=o + ox] += wv * xp[unsafe_offset=i + ox]
                        else:
                            for ox in range(ox0, ox1):
                                yp[unsafe_offset=o + ox] += wv * xp[unsafe_offset=i + ox * stride]
        if relu:
            for i in range(ho * wo):
                if yp[unsafe_offset=obase + i] < 0.0:
                    yp[unsafe_offset=obase + i] = 0.0

    # `parallelize` costs a fixed ~200 us (nn/primitives/conv2d.mojo): small layers stay on this thread
    if cw.cout * cin * k * k * ho * wo < (1 << 21):
        for co in range(cw.cout):
            _channel(co)
    else:
        parallelize(_channel, cw.cout)
    return y^


def cosine(a: List[Float32], b: List[Float32]) -> Float64:
    """The cosine of two embeddings: how alike two voices are."""
    var ab = 0.0
    var aa = 0.0
    var bb = 0.0
    for i in range(min(len(a), len(b))):
        ab += Float64(a[i]) * Float64(b[i])
        aa += Float64(a[i]) * Float64(a[i])
        bb += Float64(b[i]) * Float64(b[i])
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return ab / sqrt(aa * bb)


struct SpeakerNet(Movable):
    var stem: ConvW
    var c1: List[ConvW]
    var c2: List[ConvW]
    var sc: List[ConvW]
    """One entry per block; `cout == 0` where the block has no shortcut conv."""
    var block_stride: List[Int]
    var stage_end: List[Int]
    """The index of each stage's last block."""
    var seg_w: List[Float32]
    var seg_b: List[Float32]
    var mean_vec: List[Float32]
    var pool_eps: Float64
    var debug: Int
    """Mutants for the gate: 1 = the second stage keeps stride 1 in its
    shortcut's place (the 1x1 is applied at stride 1 and cropped), 2 = the
    pooled deviation divides by n, 4 = the pooled means and deviations change
    places. (`mean_vec` cannot be a mutant: it is all zeros in the published
    file, so leaving it in changes nothing.)"""

    def __init__(out self, var stem: ConvW):
        self.stem = stem^
        self.c1 = List[ConvW]()
        self.c2 = List[ConvW]()
        self.sc = List[ConvW]()
        self.block_stride = List[Int]()
        self.stage_end = List[Int]()
        self.seg_w = List[Float32]()
        self.seg_b = List[Float32]()
        self.mean_vec = List[Float32]()
        self.pool_eps = 1e-7
        self.debug = 0

    @staticmethod
    def _conv(ref st: SafeTensors, name: String) raises -> ConvW:
        var shape = st.shape(name + ".w")
        if len(shape) != 4 or shape[2] != shape[3]:
            raise Error("speaker: '" + name + ".w' is not a square convolution")
        return ConvW(shape[0], shape[1], shape[2], st.read_f32(name + ".w"), st.read_f32(name + ".b"))

    @staticmethod
    def load(path: String) raises -> SpeakerNet:
        var st = SafeTensors(path)
        var net = SpeakerNet(SpeakerNet._conv(st, String("stem")))
        var blocks: List[Int] = [3, 4, 6, 3]
        for s in range(SPK_STAGES):
            for b in range(blocks[s]):
                var name = "s" + String(s + 1) + ".b" + String(b)
                net.c1.append(SpeakerNet._conv(st, name + ".c1"))
                net.c2.append(SpeakerNet._conv(st, name + ".c2"))
                if st.has(name + ".sc.w"):
                    net.sc.append(SpeakerNet._conv(st, name + ".sc"))
                else:
                    net.sc.append(ConvW(0, 0, 1, List[Float32](), List[Float32]()))
                net.block_stride.append(2 if (b == 0 and s > 0) else 1)
            net.stage_end.append(len(net.c1) - 1)
        net.seg_w = st.read_f32(String("seg.w"))
        net.seg_b = st.read_f32(String("seg.b"))
        net.mean_vec = st.read_f32(String("mean_vec"))
        if len(net.seg_w) != SPK_EMB * 5120 or len(net.mean_vec) != SPK_EMB:
            raise Error("speaker: the linear layer is not 5120 -> 256")
        var eps = st.metadata(String("pool_eps"))
        if eps != "":
            net.pool_eps = Float64(eps)
        return net^

    def stem_out(self, ref feats: List[Float32]) raises -> Plane:
        """The features `[T, 80]` as the picture `[1, 80, T]`, through the stem."""
        var t = len(feats) // SPK_MELS
        if t < 8 or len(feats) != t * SPK_MELS:
            raise Error("speaker: features must be [T, 80] with T >= 8, got " + String(len(feats)) + " values")
        var x = Plane(1, SPK_MELS, t)
        for f in range(SPK_MELS):
            for i in range(t):
                x.v[f * t + i] = feats[i * SPK_MELS + f]
        return conv(x, self.stem, 1, True)

    def stage(self, var x: Plane, s: Int) -> Plane:
        """Stage `s` (0..3) on the previous stage's output."""
        var b0 = 0 if s == 0 else self.stage_end[s - 1] + 1
        for b in range(b0, self.stage_end[s] + 1):
            var stride = self.block_stride[b]
            var y = conv(x, self.c1[b], stride, True)
            var z = conv(y, self.c2[b], 1, False)
            if self.sc[b].cout > 0:
                var sc_stride = stride
                if self.debug & 1 != 0 and s == 1:
                    sc_stride = 1
                var r = conv(x, self.sc[b], sc_stride, False)
                for co in range(z.c):
                    for oy in range(z.h):
                        for ox in range(z.w):
                            z.v[(co * z.h + oy) * z.w + ox] += r.v[(co * r.h + oy) * r.w + ox]
            else:
                for i in range(len(z.v)):
                    z.v[i] += x.v[i]
            for i in range(len(z.v)):
                if z.v[i] < 0.0:
                    z.v[i] = 0.0
            x = z^
        return x^

    def pool(self, ref x: Plane) -> List[Float32]:
        """`[c, h, w]` -> per (channel, frequency) row, the mean over time,
        then the standard deviations: `2 c h` values."""
        var rows = x.c * x.h
        var out = List[Float32](length=2 * rows, fill=0.0)
        var n = Float64(x.w)
        for r in range(rows):
            var m = 0.0
            for i in range(x.w):
                m += Float64(x.v[r * x.w + i])
            m /= n
            var ss = 0.0
            for i in range(x.w):
                var d = Float64(x.v[r * x.w + i]) - m
                ss += d * d
            var div = n if self.debug & 2 != 0 else n - 1.0
            var at_mean = rows + r if self.debug & 4 != 0 else r
            var at_dev = r if self.debug & 4 != 0 else rows + r
            out[at_mean] = Float32(m)
            out[at_dev] = Float32(sqrt(ss / div + self.pool_eps))
        return out^

    def seg(self, ref pooled: List[Float32]) -> List[Float32]:
        var out = List[Float32](length=SPK_EMB, fill=0.0)
        var n = len(pooled)
        for o in range(SPK_EMB):
            var acc = Float64(self.seg_b[o])
            for i in range(n):
                acc += Float64(self.seg_w[o * n + i]) * Float64(pooled[i])
            out[o] = Float32(acc - Float64(self.mean_vec[o]))
        return out^

    def embed(self, ref feats: List[Float32]) raises -> List[Float32]:
        """Mean-subtracted features `[T, 80]` -> the 256-value embedding."""
        var x = self.stem_out(feats)
        for s in range(SPK_STAGES):
            x = self.stage(x^, s)
        var p = self.pool(x)
        return self.seg(p)
