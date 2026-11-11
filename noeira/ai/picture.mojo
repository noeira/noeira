# +--------------------------------------------------------------------------+ #
# | noeira.ai.picture — an RGB frame, as a decision service takes it
# +--------------------------------------------------------------------------+ #
"""`Picture`: an RGB24 frame, `rgb[(y W + x) 3 + c]` (HWC, uint8) — what
`noeira.vision.camera_thread` hands out with `rgb=True`, and what
`DecisionService.start_seeing` and LFM3-VL's tower (`noeira.llm.lfm3.vision`)
take. Light on purpose: the trait's users do not import a model.

`to_bytes` / `picture_from` are the framing `D1Service` sends them in:
`[u32 height][u32 width][H W 3 bytes]`, little-endian.
"""

from std.memory import unsafe_memcpy


struct Picture(Copyable, Movable):
    var rgb: List[UInt8]
    var height: Int
    var width: Int

    def __init__(out self, var rgb: List[UInt8], height: Int, width: Int) raises:
        if height <= 0 or width <= 0 or len(rgb) != height * width * 3:
            raise Error("picture: " + String(len(rgb)) + " bytes for " + String(height) + "x" + String(width) + "x3")
        self.rgb = rgb^
        self.height = height
        self.width = width

    def framed_bytes(self) -> Int:
        return 8 + len(self.rgb)

    def append_to(self, mut out: List[UInt8]):
        """`[u32 height][u32 width][rgb]` at the end of `out` (one copy)."""
        _put_u32(out, self.height)
        _put_u32(out, self.width)
        var at = len(out)
        out.resize(at + len(self.rgb), 0)
        unsafe_memcpy(dest=out.unsafe_ptr().unsafe_offset(at), src=self.rgb.unsafe_ptr(), count=len(self.rgb))


def picture_from(ref b: List[UInt8], mut at: Int) raises -> Picture:
    """The picture framed at `b[at:]` (`Picture.append_to`); `at` moves past it."""
    if at + 8 > len(b):
        raise Error("picture: a truncated header")
    var h = _get_u32(b, at)
    var w = _get_u32(b, at + 4)
    var n = h * w * 3
    if h <= 0 or w <= 0 or at + 8 + n > len(b):
        raise Error("picture: " + String(h) + "x" + String(w) + " does not fit the " + String(len(b) - at - 8) + " bytes left")
    var rgb = List[UInt8](unsafe_uninit_length=n)
    unsafe_memcpy(dest=rgb.unsafe_ptr(), src=b.unsafe_ptr().unsafe_offset(at + 8), count=n)
    at += 8 + n
    return Picture(rgb^, h, w)


def _put_u32(mut out: List[UInt8], v: Int):
    for k in range(4):
        out.append(UInt8((v >> (8 * k)) & 0xFF))


def _get_u32(ref b: List[UInt8], at: Int) -> Int:
    var v = 0
    for k in range(4):
        v |= Int(b[at + k]) << (8 * k)
    return v
