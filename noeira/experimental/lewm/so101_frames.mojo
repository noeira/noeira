"""Camera frames for the SO-101 world model: one resize for sim and real.

noeira-docs/SO101_LEWM_PLAN.md S0 / S2. The rig's cameras are 320 × 240
(the sim tracer, `tower_demo_rerender.mojo`) and the real teleop imports
store the same size undistorted to the sim pinhole. The model sees R × R
(112 by default). Sim renders and real frames MUST go through the same
function, or S2's real-vs-sim comparison measures the resampler.

`area_resize_chw` is an exact area average: output pixel o on an axis covers
the source interval [o·s, (o+1)·s), s = n_in / n_out, and each source pixel
contributes its overlap / s. Separable (x then y), sparse (each output reads
⌈s⌉ + 1 source pixels per axis). 320 × 240 -> R × R squashes the 4:3 aspect
on purpose: the ViT is square, and the squash is the same for both domains.
"""

from std.math import floor


@fieldwise_init
struct AreaAxis(Copyable, Movable):
    """The sparse weights of one axis: output o reads source pixels
    `idx[start[o] ..< start[o + 1]]` with weights `w[...]` (summing to 1)."""

    var start: List[Int]
    var idx: List[Int]
    var w: List[Float64]

    @staticmethod
    def make(n_in: Int, n_out: Int) -> Self:
        var s = Float64(n_in) / Float64(n_out)
        var start = List[Int](capacity=n_out + 1)
        var idx = List[Int]()
        var w = List[Float64]()
        for o in range(n_out):
            start.append(len(idx))
            var lo = Float64(o) * s
            var hi = Float64(o + 1) * s
            var i = Int(floor(lo))
            while Float64(i) < hi and i < n_in:
                var ov = min(hi, Float64(i + 1)) - max(lo, Float64(i))
                if ov > 1e-12:
                    idx.append(i)
                    w.append(ov / s)
                i += 1
        start.append(len(idx))
        return Self(start^, idx^, w^)


struct AreaResize(Movable):
    """(C, H, W) u8 frames -> (C, R, R) u8, exact area average."""

    var C: Int
    var H: Int
    var W: Int
    var R: Int
    var ax: AreaAxis
    var ay: AreaAxis

    def __init__(out self, C: Int, H: Int, W: Int, R: Int):
        self.C = C
        self.H = H
        self.W = W
        self.R = R
        self.ax = AreaAxis.make(W, R)
        self.ay = AreaAxis.make(H, R)

    def frame(
        self, src: List[UInt8], src_off: Int, mut dst: List[UInt8], dst_off: Int
    ):
        """One (C, H, W) frame at `src_off` -> (C, R, R) at `dst_off`."""
        var tmp = List[Float64](length=self.H * self.R, fill=0.0)
        for c in range(self.C):
            var base = src_off + c * self.H * self.W
            for y in range(self.H):
                var row = base + y * self.W
                for o in range(self.R):
                    var acc = 0.0
                    for k in range(self.ax.start[o], self.ax.start[o + 1]):
                        acc += self.ax.w[k] * Float64(Int(src[row + self.ax.idx[k]]))
                    tmp[y * self.R + o] = acc
            var out = dst_off + c * self.R * self.R
            for oy in range(self.R):
                for ox in range(self.R):
                    var acc = 0.0
                    for k in range(self.ay.start[oy], self.ay.start[oy + 1]):
                        acc += self.ay.w[k] * tmp[self.ay.idx[k] * self.R + ox]
                    var v = Int(acc + 0.5)
                    dst[out + oy * self.R + ox] = UInt8(min(255, max(0, v)))

    def frames(self, src: List[UInt8], n: Int) -> List[UInt8]:
        """`n` consecutive frames."""
        var fin = self.C * self.H * self.W
        var fout = self.C * self.R * self.R
        var dst = List[UInt8](length=n * fout, fill=UInt8(0))
        for i in range(n):
            self.frame(src, i * fin, dst, i * fout)
        return dst^
