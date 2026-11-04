"""Dreamer 4 tokenizer lighthouse — GPU (Pong reconstruction).

    pixi run -e apple  mojo run -I . examples/dreamer4/pong_tokenizer_lighthouse_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/dreamer4/pong_tokenizer_lighthouse_gpu.mojo

Same as the CPU lighthouse but the tokenizer trains on the GPU. Data prep
(stream Pong, take latest grayscale channel, box-downscale 84→32, patchify)
stays on the host; the patches are copied H2D and the model forward/backward
+ Adam run on device. The masked-MSE training gradient is a device kernel
(`masked_recon_grad_gpu`); PSNR eval downloads `pred` and scores on host.

Gate: full-frame PSNR climbs over training, on GPU.
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.envs.arcade_games.pong.online_sampler import (
    OnlinePongSampler, ScriptedPongPolicy,
)
from noeira.data.window_source import WindowSource
from noeira.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from noeira.deep_agents.dreamer4.recon_loss import (
    masked_recon_grad_gpu, full_recon_psnr,
)
from noeira.deep_agents.dreamer4.patchify import downscale_box, temporal_patchify
from noeira.deep_agents.dreamer4.shortcut_loss import _mao


def main() raises:
    print("=" * 70)
    print("Dreamer 4 tokenizer lighthouse — Pong reconstruction (GPU)")
    print("=" * 70)

    comptime IN_CH = 4
    comptime IMG = 84
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime ACT = 3
    comptime T = 4
    comptime B = 4
    comptime BATCH = B * T

    comptime TGT = 32
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 16
    comptime DP = PATCH * PATCH                      # 64 (C=1)
    comptime D = 64
    comptime NH = 4
    comptime L = 8
    comptime D_BOT = 16
    comptime HID = 256
    comptime DEPTH = 2
    comptime DROP = 0.5
    comptime STEPS = 150
    comptime EVAL_EVERY = 25
    comptime LR = Scalar[DT](2e-3)

    comptime FRAME_N = BATCH * TGT * TGT
    comptime PATCH_N = BATCH * NP * DP

    var ctx = DeviceContext()
    comptime OnlineBuf = OnlinePongSampler[ScriptedPongPolicy, B, T]
    var src = WindowSource[IMG_DIM, ACT, T, B, "cpu", OnlineBuf].make(
        OnlineBuf.make(ScriptedPongPolicy(eps=0.3))
    )

    var tok = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, DROP, DROP, 7
    ].make["gpu", Xavier](ctx)
    var optim = Adam(lr=LR)

    # host data-prep buffers (CPU-side `Tensor`s; patches uploaded H2D each step)
    var frames = Tensor.alloc(FRAME_N)
    var patches = Tensor.alloc(PATCH_N)         # host staging; .upload() → device
    var pred = Tensor.alloc_gpu(ctx, PATCH_N)
    var gpred = Tensor.alloc_gpu(ctx, PATCH_N)
    var gin = Tensor.alloc_gpu(ctx, PATCH_N)

    var first_psnr: Float64 = 0.0
    var last_psnr: Float64 = 0.0

    for step in range(STEPS):
        src.next_batch()
        var pix = src.pix_ptr()
        for b in range(B):
            for t in range(T):
                var bt = b * T + t
                var fsrc = pix + (b * T + t) * IMG_DIM + 3 * IMG * IMG
                downscale_box[IMG, IMG, TGT, TGT](
                    fsrc, _mao(frames.data.unsafe_ptr()) + bt * TGT * TGT
                )
        temporal_patchify[BATCH, 1, TGT, TGT, PATCH](
            _mao(frames.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr())
        )
        patches.upload(ctx)                      # H2D the freshly-patchified batch

        optim.zero_grad["gpu"](tok, ctx)
        tok.forward["gpu", BATCH](TensorRefs[1](patches), pred, ctx)
        var mask = tok.mae_mask_ptr()  # device keep ptr
        masked_recon_grad_gpu[NP, DP, BATCH](
            _mao(pred.dev.value().unsafe_ptr()),
            _mao(patches.dev.value().unsafe_ptr()),
            mask,
            _mao(gpred.dev.value().unsafe_ptr()),
            ctx,
        )
        tok.vjp["gpu", BATCH](TensorRefs[1](patches), gpred, TensorRefs[1](gin), ctx)
        optim.step["gpu"](tok, ctx)
        tok.advance_rng()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            tok.set_mae_p(0.0, 0.0)
            tok.forward["gpu", BATCH](TensorRefs[1](patches), pred, ctx)
            pred.download(ctx)                   # D2H for the host PSNR score
            var psnr = full_recon_psnr[NP, DP, BATCH](
                _mao(pred.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr())
            )
            tok.set_mae_p(DROP, DROP)
            if step == 0:
                first_psnr = psnr
            last_psnr = psnr
            print("  step", step, " full PSNR =", psnr, "dB")

    print("-" * 70)
    print("  first PSNR =", first_psnr, "dB   final PSNR =", last_psnr, "dB")
    assert_true(last_psnr > first_psnr + 3.0, "PSNR must climb (>3 dB) on GPU")
    print("=" * 70)
    print("GPU LIGHTHOUSE PASSED — tokenizer learns Pong reconstruction on GPU")
    print("=" * 70)
