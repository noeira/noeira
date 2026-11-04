"""WindowSource — bridge any `OfflineBuffer` to a pixel world-model trainer.

Generic over the data source: it samples length-T windows from any
`noeira.core.offline_buffer.OfflineBuffer` (a loaded `PongOfflineBuffer`,
the live `OnlinePongSampler`, …) and hands the trainer exactly what
`train_step` consumes — fp32 pixels `(B, T·IMG_DIM)` (CHW, ÷255) and fp32
one-hot actions `(B, T·ACT)` — on the chosen target. Dreamer 4's Pong
lighthouses train on it.

`target` and `BUF` are required: `WindowSource[IMG_DIM, ACT, T, B, "gpu",
PongOfflineBuffer]`, or `OnlinePongSampler[…]` to stream live simulator
windows with no other change to the trainer / convert kernel / loop.

**CHW only (for now).** The bridge normalises in place via
`u8_to_fp32_norm` (layout-preserving), so it requires
`BUF.INPUT_LAYOUT_HWC == False` (enforced by a comptime assert). HWC buffers
(e.g. PushT) need the `u8_hwc_to_chw_norm` permute branch — a small TODO.

Flow per batch:
  buf.sample_batch_uint8 → host uint8 pix + host fp32 one-hot act
  CPU:  convert uint8→fp32 (host)              → pix/act host ptrs
  GPU:  H2D uint8 + act, convert uint8→fp32 dev → pix/act device ptrs
The trainer wraps `pix_ptr`/`act_ptr` in `TileTensor`s and calls train_step.
"""

from std.memory import alloc
from noeira.nn.core.ptr import untracked
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.data.pixels import u8_to_fp32_norm, u8_hwc_to_chw_norm
from noeira.core.offline_buffer import OfflineBuffer


struct WindowSource[
    IMG_DIM: Int,
    ACT: Int,
    T: Int,
    B: Int,
    target: StaticString,
    BUF: OfflineBuffer,
    C: Int = 0,
    FRAME: Int = 0,
](Movable & Deinitable):
    # `C`/`FRAME` are the per-frame channel count + side length, required ONLY
    # when `BUF.INPUT_LAYOUT_HWC` (e.g. PushT 3×224×224): then conversion is
    # `u8_hwc_to_chw_norm` (permute HWC→CHW + ÷255). CHW buffers (Pong) leave
    # them 0 and use the layout-preserving `u8_to_fp32_norm`.
    comptime NPIX = Self.B * Self.T * Self.IMG_DIM
    comptime NACT = Self.B * Self.T * Self.ACT

    var buf: Self.BUF
    # Host staging (always): sampled uint8 pixels + fp32 one-hot actions.
    var pix_u8_host: Pointer[Scalar[DType.uint8], MutUntrackedOrigin]
    var act_host: Pointer[Scalar[DT], MutUntrackedOrigin]
    # CPU output: converted fp32 pixels (host). GPU: device buffers below.
    var pix_fp32_host: Pointer[Scalar[DT], MutUntrackedOrigin]
    var pix_u8_dev: Optional[DeviceBuffer[DType.uint8]]
    var pix_fp32_dev: Optional[DeviceBuffer[DT]]
    var act_dev: Optional[DeviceBuffer[DT]]
    var ctx: Optional[DeviceContext]

    def __init__(out self, var buf: Self.BUF):
        comptime if Self.BUF.INPUT_LAYOUT_HWC:
            comptime assert (
                Self.C > 0
                and Self.C * Self.FRAME * Self.FRAME == Self.IMG_DIM
            ), (
                "WindowSource: HWC buffer (INPUT_LAYOUT_HWC=True) needs C/FRAME"
                " params with C*FRAME*FRAME == IMG_DIM (e.g. C=3, FRAME=224)."
            )
        self.buf = buf^
        self.pix_u8_host = untracked(
            alloc[Scalar[DType.uint8]]({count = Self.NPIX}).unsafe_leak()
        )
        self.act_host = untracked(
            alloc[Scalar[DT]]({count = Self.NACT}).unsafe_leak()
        )
        self.pix_fp32_host = untracked(
            alloc[Scalar[DT]]({count = Self.NPIX}).unsafe_leak()
        )
        self.pix_u8_dev = None
        self.pix_fp32_dev = None
        self.act_dev = None
        self.ctx = None

    def __deinit__(deinit self):
        self.pix_u8_host.unsafe_free()
        self.act_host.unsafe_free()
        self.pix_fp32_host.unsafe_free()

    @staticmethod
    def make(
        var buf: Self.BUF,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var s = Self(buf^)
        s.ctx = ctx
        comptime if Self.target == "gpu":
            if not ctx:
                raise Error("WindowSource.make[gpu]: ctx required")
            var c = ctx.value()
            s.pix_u8_dev = c.enqueue_create_buffer[DType.uint8](Self.NPIX)
            s.pix_fp32_dev = c.enqueue_create_buffer[DT](Self.NPIX)
            s.act_dev = c.enqueue_create_buffer[DT](Self.NACT)
        return s^

    def next_batch(mut self) raises:
        """Sample one (B, T) window batch and populate the target buffers.
        After this call `pix_ptr()` / `act_ptr()` are valid for one
        `train_step` / `eval_loss`."""
        self.buf.sample_batch_uint8(
            Self.B, Self.T, self.pix_u8_host.as_unsafe_any_origin(), self.act_host.as_unsafe_any_origin()
        )
        comptime if Self.target == "cpu":
            comptime if Self.BUF.INPUT_LAYOUT_HWC:
                u8_hwc_to_chw_norm[
                    "cpu", Self.C, Self.FRAME, Self.FRAME, Self.B * Self.T
                ](self.pix_u8_host, self.pix_fp32_host)
            else:
                u8_to_fp32_norm["cpu", Self.NPIX](
                    self.pix_u8_host, self.pix_fp32_host
                )
        else:
            var c = self.ctx.value()
            c.enqueue_copy(self.pix_u8_dev.value(), self.pix_u8_host)
            c.enqueue_copy(self.act_dev.value(), self.act_host)
            var src_u8 = rebind[
                Pointer[Scalar[DType.uint8], MutAnyOrigin]
            ](self.pix_u8_dev.value().unsafe_ptr())
            var dst_fp32 = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                self.pix_fp32_dev.value().unsafe_ptr()
            )
            comptime if Self.BUF.INPUT_LAYOUT_HWC:
                u8_hwc_to_chw_norm[
                    "gpu", Self.C, Self.FRAME, Self.FRAME, Self.B * Self.T
                ](src_u8, dst_fp32, ctx=self.ctx)
            else:
                u8_to_fp32_norm["gpu", Self.NPIX](
                    src_u8, dst_fp32, ctx=self.ctx
                )

    def pix_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        comptime if Self.target == "cpu":
            return self.pix_fp32_host.as_unsafe_any_origin()
        else:
            return rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                self.pix_fp32_dev.value().unsafe_ptr()
            )

    def act_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        comptime if Self.target == "cpu":
            return self.act_host.as_unsafe_any_origin()
        else:
            return rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                self.act_dev.value().unsafe_ptr()
            )
