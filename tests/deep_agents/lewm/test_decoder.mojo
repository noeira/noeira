"""LeWM decoder probe — patchify round-trip + overfit.

  - patchify∘unpatchify == identity (CHW ↔ patch-major), CPU + GPU.
  - LeWMDecoderTrainer overfits a FIXED (emb → target image) pair: with
    learnable queries + global conditioning + residual-MLP layers the loss
    must collapse, proving forward/backward/optimizer wire end to end.

Run:  pixi run -e apple mojo run -I . tests/deep_agents/lewm/test_decoder.mojo
"""

from std.memory import alloc
from max.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.deep_agents.lewm.decoder import patchify, unpatchify
from noeira.deep_agents.lewm.decoder_trainer import LeWMDecoderTrainer


# toy dims
comptime C = 2
comptime IMG = 4
comptime PATCH_D = 2
comptime GRID = IMG // PATCH_D       # 2
comptime N_Q = GRID * GRID           # 4
comptime PATCH_PX = C * PATCH_D * PATCH_D   # 8
comptime IMGN = C * IMG * IMG        # 32  (== N_Q * PATCH_PX)

comptime EMB = 6
comptime HID = 8
comptime FF = 16
comptime N_LAYERS = 2
comptime BATCH = 3


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n).as_unsafe_any_origin()


def _p(b: DeviceBuffer[DT]) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int) -> Scalar[DT]:
    return Scalar[DT]((Float64((i * 2654435761) % 1000) / 500.0) - 1.0)


def test_patchify_roundtrip_cpu() raises:
    print("patchify roundtrip cpu ...")
    var img = _a(BATCH * IMGN)
    var pat = _a(BATCH * N_Q * PATCH_PX)
    var back = _a(BATCH * IMGN)
    for k in range(BATCH * IMGN):
        img[k] = _det(k + 3)
    patchify["cpu", BATCH, C, IMG, PATCH_D](None, img, pat)
    unpatchify["cpu", BATCH, C, IMG, PATCH_D](None, pat, back)
    var maxd: Scalar[DT] = 0.0
    for k in range(BATCH * IMGN):
        var d = (img[k] - back[k]).__abs__()
        if d > maxd:
            maxd = d
    assert_true(maxd < Scalar[DT](1e-7), "patchify∘unpatchify == identity (cpu)")
    img.free(); pat.free(); back.free()
    print("  ok")


def test_patchify_roundtrip_gpu() raises:
    print("patchify roundtrip gpu ...")
    var ctx = DeviceContext()
    var imgd = ctx.enqueue_create_buffer[DT](BATCH * IMGN)
    var patd = ctx.enqueue_create_buffer[DT](BATCH * N_Q * PATCH_PX)
    var backd = ctx.enqueue_create_buffer[DT](BATCH * IMGN)
    var imgh = ctx.enqueue_create_host_buffer[DT](BATCH * IMGN)
    var backh = ctx.enqueue_create_host_buffer[DT](BATCH * IMGN)
    ctx.synchronize()
    for k in range(BATCH * IMGN):
        imgh.unsafe_ptr()[k] = _det(k + 3)
    ctx.enqueue_copy(imgd, imgh); ctx.synchronize()
    patchify["gpu", BATCH, C, IMG, PATCH_D](ctx, _p(imgd), _p(patd))
    unpatchify["gpu", BATCH, C, IMG, PATCH_D](ctx, _p(patd), _p(backd))
    ctx.enqueue_copy(backh, backd); ctx.synchronize()
    var maxd: Scalar[DT] = 0.0
    for k in range(BATCH * IMGN):
        var d = (imgh.unsafe_ptr()[k] - backh.unsafe_ptr()[k]).__abs__()
        if d > maxd:
            maxd = d
    assert_true(maxd < Scalar[DT](1e-7), "patchify∘unpatchify == identity (gpu)")
    print("  ok")


def test_overfit_cpu() raises:
    print("decoder overfit cpu ...")
    var tr = LeWMDecoderTrainer[
        EMB, HID, N_Q, PATCH_PX, FF, N_LAYERS, BATCH, "cpu"
    ].make(lr=Scalar[DT](3e-3))

    var emb = _a(BATCH * EMB)
    var img = _a(BATCH * IMGN)
    var tgt = _a(BATCH * N_Q * PATCH_PX)
    for k in range(BATCH * EMB):
        emb[k] = _det(k + 11)
    for k in range(BATCH * IMGN):
        img[k] = _det(k + 200)
    patchify["cpu", BATCH, C, IMG, PATCH_D](None, img, tgt)
    var emb_t = TileTensor(emb, row_major[BATCH, EMB]())
    var tgt_t = TileTensor(tgt, row_major[BATCH, N_Q * PATCH_PX]())

    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    for step in range(300):
        var l = tr.train_step(emb_t, tgt_t)
        if step == 0:
            first = l
        last = l
    print("   loss", first, "→", last)
    assert_true(last < first * Scalar[DT](0.2), "decoder overfits (cpu)")
    emb.free(); img.free(); tgt.free()
    _ = tr^
    print("  ok")


def test_overfit_gpu() raises:
    print("decoder overfit gpu ...")
    var ctx = DeviceContext()
    var tr = LeWMDecoderTrainer[
        EMB, HID, N_Q, PATCH_PX, FF, N_LAYERS, BATCH, "gpu"
    ].make(lr=Scalar[DT](3e-3), ctx=ctx)

    var embd = ctx.enqueue_create_buffer[DT](BATCH * EMB)
    var imgd = ctx.enqueue_create_buffer[DT](BATCH * IMGN)
    var tgtd = ctx.enqueue_create_buffer[DT](BATCH * N_Q * PATCH_PX)
    var embh = ctx.enqueue_create_host_buffer[DT](BATCH * EMB)
    var imgh = ctx.enqueue_create_host_buffer[DT](BATCH * IMGN)
    ctx.synchronize()
    for k in range(BATCH * EMB):
        embh.unsafe_ptr()[k] = _det(k + 11)
    for k in range(BATCH * IMGN):
        imgh.unsafe_ptr()[k] = _det(k + 200)
    ctx.enqueue_copy(embd, embh); ctx.enqueue_copy(imgd, imgh); ctx.synchronize()
    patchify["gpu", BATCH, C, IMG, PATCH_D](ctx, _p(imgd), _p(tgtd))

    var emb_t = TileTensor(_p(embd), row_major[BATCH, EMB]())
    var tgt_t = TileTensor(_p(tgtd), row_major[BATCH, N_Q * PATCH_PX]())

    tr.reset_loss_accum()
    for _ in range(150):
        _ = tr.train_step(emb_t, tgt_t)
    var early = tr.read_loss_accum()
    tr.reset_loss_accum()
    for _ in range(150):
        _ = tr.train_step(emb_t, tgt_t)
    var late = tr.read_loss_accum()
    print("   mean loss window1", early, " window2", late)
    assert_true(late < early * Scalar[DT](0.5), "decoder overfits (gpu)")
    _ = tr^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM decoder probe — patchify + overfit")
    print("=" * 70)
    test_patchify_roundtrip_cpu()
    test_patchify_roundtrip_gpu()
    test_overfit_cpu()
    test_overfit_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
