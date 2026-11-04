"""WindowSource HWC branch (CPU + GPU, mock buffer — no PushT data).

PushT's `PushTOfflineSampler` delivers HWC uint8 (INPUT_LAYOUT_HWC=True).
This test stands up a tiny mock HWC `OfflineBuffer` and checks the
WindowSource HWC path permutes + normalises correctly:
  dst_CHW[f, c, h, w] == src_HWC[f, h, w, c] / 255   (every element)
on both CPU and GPU — validating the bridge without the 13 GB download.

Run:  pixi run -e apple mojo run -I . tests/data/test_window_source_hwc.mojo
"""

from std.memory import alloc
from max.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.core.offline_buffer import OfflineBuffer
from noeira.data.window_source import WindowSource


comptime C = 3
comptime FRAME = 4
comptime IMG_DIM = C * FRAME * FRAME   # 48
comptime ACT = 3
comptime T = 2
comptime B = 2
comptime NPIX = B * T * IMG_DIM


# Mock HWC source: fills a deterministic HWC byte pattern + one-hot actions.
struct MockHWC(Movable, OfflineBuffer):
    comptime INPUT_LAYOUT_HWC: Bool = True

    def __init__(out self):
        pass

    def sample_batch_uint8(
        mut self, B_: Int, T_: Int,
        pixels_out: Pointer[Scalar[DType.uint8], MutAnyOrigin],
        actions_out: Pointer[Scalar[DType.float32], MutAnyOrigin],
    ) raises:
        for i in range(B_ * T_ * IMG_DIM):
            pixels_out[i] = UInt8((i * 37 + 11) % 256)
        for b in range(B_):
            for t in range(T_):
                for a in range(ACT):
                    actions_out[(b * T_ + t) * ACT + a] = Float32(
                        1.0 if a == ((b + t) % ACT) else 0.0
                    )


def _byte(i: Int) -> Float64:
    return Float64((i * 37 + 11) % 256)


def _check(pix: Pointer[Scalar[DT], MutAnyOrigin]) raises:
    # dst is CHW: dst[f, c, h, w] @ f*IMG_DIM + (c*FRAME+h)*FRAME + w
    # src is HWC: src[f, h, w, c] @ f*IMG_DIM + (h*FRAME+w)*C + c
    var bad = 0
    for f in range(B * T):
        for c in range(C):
            for h in range(FRAME):
                for w in range(FRAME):
                    var dst_i = f * IMG_DIM + (c * FRAME + h) * FRAME + w
                    var src_i = f * IMG_DIM + (h * FRAME + w) * C + c
                    var want = _byte(src_i) / 255.0
                    if (Float64(pix[dst_i]) - want).__abs__() > 1e-6:
                        bad += 1
    assert_true(bad == 0, "HWC→CHW permute+÷255 correct for every element")


def main() raises:
    print("=" * 70)
    print("WindowSource HWC branch (mock buffer)")
    print("=" * 70)

    # ── CPU
    print("cpu ...")
    var src_c = WindowSource[
        IMG_DIM, ACT, T, B, "cpu", MockHWC, C, FRAME
    ].make(MockHWC())
    src_c.next_batch()
    _check(src_c.pix_ptr())
    print("   cpu ok")

    # ── GPU
    print("gpu ...")
    var ctx = DeviceContext()
    var src_g = WindowSource[
        IMG_DIM, ACT, T, B, "gpu", MockHWC, C, FRAME
    ].make(MockHWC(), ctx=ctx)
    src_g.next_batch()
    var host = alloc[Scalar[DT]](NPIX).as_unsafe_any_origin()
    var dev = DeviceBuffer[DT](ctx, src_g.pix_ptr(), NPIX, owning=False)
    ctx.enqueue_copy(host, dev)
    ctx.synchronize()
    _check(host.as_unsafe_any_origin())
    host.free()
    print("   gpu ok")

    _ = src_c^
    _ = src_g^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
