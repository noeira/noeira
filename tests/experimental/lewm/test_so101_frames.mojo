"""`so101_frames.AreaResize`: the one resampler for sim and real SO-101 frames.

    pixi run mojo run -I . tests/experimental/lewm/test_so101_frames.mojo

1. a constant frame stays that constant at any size;
2. an integer ratio (320 -> 160 on x) reads source pixels 2o and 2o + 1 at
   weight 1/2 each;
3. the 4:3 squash to 112 × 112 keeps the frame's mean to within rounding
   (the area weights sum to 1 on both axes).
"""

from std.random import seed, random_ui64

from noeira.experimental.lewm.so101_frames import AreaResize


def main() raises:
    var fails = 0
    var C = 3
    var H = 240
    var W = 320
    var src = List[UInt8](length=C * H * W, fill=UInt8(173))
    var r1 = AreaResize(C, H, W, 112).frames(src, 1)
    var ok1 = True
    for v in r1:
        if v != UInt8(173):
            ok1 = False
    print("  1. constant 173 -> 112x112 constant:", ok1)
    if not ok1:
        fails += 1

    seed(5)
    for i in range(len(src)):
        src[i] = UInt8(Int(random_ui64(0, 255)))
    # 320 -> 160 is an exact 2:1 ratio on x: check that axis's weights
    var wx = AreaResize(C, H, W, 160).ax.copy()
    var ok2 = True
    for o in range(160):
        var n = wx.start[o + 1] - wx.start[o]
        if n != 2 or abs(wx.w[wx.start[o]] - 0.5) > 1e-12 or wx.idx[wx.start[o]] != 2 * o:
            ok2 = False
    print("  2. 320 -> 160: every output averages source pixels 2o, 2o+1 at 1/2:", ok2)
    if not ok2:
        fails += 1

    var r3 = AreaResize(C, H, W, 112).frames(src, 1)
    var m_in = 0.0
    for v in src:
        m_in += Float64(Int(v))
    m_in /= Float64(len(src))
    var m_out = 0.0
    for v in r3:
        m_out += Float64(Int(v))
    m_out /= Float64(len(r3))
    var ok3 = abs(m_in - m_out) < 0.5
    print("  3. random frame mean", m_in, "->", m_out, "| within 0.5:", ok3)
    if not ok3:
        fails += 1
    if fails > 0:
        raise Error("FAIL: " + String(fails))
    print("PASS")
