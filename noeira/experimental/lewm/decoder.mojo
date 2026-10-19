"""LeWM decoder — lightweight transformer reconstruction probe (viz only).

The LeWM JEPA has NO decoder by design; this is a diagnostic head trained
on the **frozen** encoder to visualize what the global representation
retains (paper §"Decoder (Visualization Only)"). It decodes the global
embedding (our pooled `emb`, the analog of the paper's [CLS] token) into an
image via learnable per-patch query tokens that cross-attend to the global
vector through several residual-MLP layers, then linearly project to pixel
patches that are un-patchified to an RGB image.

  emb (B, EMB)
    ├─ Linear[EMB,HID] ─ g ─ BroadcastTokens[N_Q] ─ gN (B, N_Q·HID)  (cond)
    └─ LearnedQueries[N_Q,HID] ─ q0 (B, N_Q·HID)                     (queries)
  xL  = RepeatConditional[N_LAYERS, DecoderBlock[N_Q,HID,FF]](q0, gN)
  xf  = Tokenwise[N_Q, LayerNorm[HID]](xL)
  recon = Tokenwise[N_Q, Linear[HID, PATCH_PX]](xf)   (B, N_Q·PATCH_PX)

`DecoderBlock` is the paper's cross-attention-to-a-single-global-token layer
in its exact mathematical equivalent (one KV token ⇒ query-independent
injection); see `nn/primitives/decoder_block.mojo`.

The decoder works in **patch space** (B, N_Q·PATCH_PX): the target image is
`patchify`-d to the same layout and the loss is per-patch MSE; for display
both `recon` and target are `unpatchify`-d back to CHW. Patch-pixel layout
inside a patch is channel-major `[c, i, j]`. `PATCH_D` (decoder patch size,
paper=16) is independent of the encoder's patch size.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT, TPB
from noeira.nn import (
    ComputeGraph,
    InputSlot,
    Node,
    Tokenwise,
    RepeatConditional,
    Linear,
    LayerNorm,
    BroadcastTokens,
    LearnedQueries,
    DecoderBlock,
    MSEPerSample,
)


# Reconstruction graph: emb → recon patches.  N_Q = (IMG//PATCH_D)^2,
# PATCH_PX = C·PATCH_D·PATCH_D (caller supplies the derived dims).
comptime LeWMDecoder[
    EMB: Int,
    HID: Int,
    N_Q: Int,
    PATCH_PX: Int,
    FF: Int,
    N_LAYERS: Int,
] = ComputeGraph[
    InputSlot["emb", EMB],
    Node["g", Linear[EMB, HID], "emb"],
    Node["gN", BroadcastTokens[N_Q, HID], "g"],
    Node["q0", LearnedQueries[EMB, N_Q, HID], "emb"],
    Node[
        "xL",
        RepeatConditional[N_LAYERS, DecoderBlock[N_Q, HID, FF]],
        "q0",
        "gN",
    ],
    Node["xf", Tokenwise[N_Q, LayerNorm[HID]], "xL"],
    Node["recon", Tokenwise[N_Q, Linear[HID, PATCH_PX]], "xf"],
]


# Training graph: adds the patchified target input + per-sample MSE.
comptime LeWMDecoderLossGraph[
    EMB: Int,
    HID: Int,
    N_Q: Int,
    PATCH_PX: Int,
    FF: Int,
    N_LAYERS: Int,
] = ComputeGraph[
    InputSlot["emb", EMB],
    InputSlot["tgt", N_Q * PATCH_PX],
    Node["g", Linear[EMB, HID], "emb"],
    Node["gN", BroadcastTokens[N_Q, HID], "g"],
    Node["q0", LearnedQueries[EMB, N_Q, HID], "emb"],
    Node[
        "xL",
        RepeatConditional[N_LAYERS, DecoderBlock[N_Q, HID, FF]],
        "q0",
        "gN",
    ],
    Node["xf", Tokenwise[N_Q, LayerNorm[HID]], "xL"],
    Node["recon", Tokenwise[N_Q, Linear[HID, PATCH_PX]], "xf"],
    Node["loss", MSEPerSample[N_Q * PATCH_PX], "recon", "tgt"],
]


# ── patchify / unpatchify (CHW image ↔ patch-major, no autodiff) ───────
# img    : (BATCH, C·IMG·IMG)  channel-major
# patches: (BATCH, N_Q·PATCH_PX)  patch-major, within-patch [c, i, j]
#   N_Q = GRID·GRID, GRID = IMG//PATCH_D, PATCH_PX = C·PATCH_D·PATCH_D


def _patchify_kernel[
    BATCH: Int, C: Int, IMG: Int, PATCH_D: Int
](
    img: Tile[DT, BATCH, C * IMG * IMG],
    dst: Tile[DT, BATCH, (IMG // PATCH_D) * (IMG // PATCH_D) * (C * PATCH_D * PATCH_D)],
):
    comptime GRID = IMG // PATCH_D
    comptime PATCH_PX = C * PATCH_D * PATCH_D
    comptime NQ = GRID * GRID
    comptime total = BATCH * NQ * PATCH_PX
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var b = idx // (NQ * PATCH_PX)
    var rem = idx % (NQ * PATCH_PX)
    var p = rem // PATCH_PX
    var inp = rem % PATCH_PX
    var c = inp // (PATCH_D * PATCH_D)
    var ij = inp % (PATCH_D * PATCH_D)
    var i = ij // PATCH_D
    var j = ij % PATCH_D
    var ph = p // GRID
    var pw = p % GRID
    var src = (
        b * (C * IMG * IMG)
        + c * (IMG * IMG)
        + (ph * PATCH_D + i) * IMG
        + (pw * PATCH_D + j)
    )
    dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](img.ptr[unsafe_offset=src])


def _unpatchify_kernel[
    BATCH: Int, C: Int, IMG: Int, PATCH_D: Int
](
    patches: Tile[DT, BATCH, (IMG // PATCH_D) * (IMG // PATCH_D) * (C * PATCH_D * PATCH_D)],
    img: Tile[DT, BATCH, C * IMG * IMG],
):
    comptime GRID = IMG // PATCH_D
    comptime PATCH_PX = C * PATCH_D * PATCH_D
    comptime total = BATCH * C * IMG * IMG
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var b = idx // (C * IMG * IMG)
    var rem = idx % (C * IMG * IMG)
    var c = rem // (IMG * IMG)
    var hw = rem % (IMG * IMG)
    var y = hw // IMG
    var x = hw % IMG
    var ph = y // PATCH_D
    var i = y % PATCH_D
    var pw = x // PATCH_D
    var j = x % PATCH_D
    var p = ph * GRID + pw
    var psrc = (
        b * (GRID * GRID * PATCH_PX)
        + p * PATCH_PX
        + c * (PATCH_D * PATCH_D)
        + i * PATCH_D
        + j
    )
    img.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](patches.ptr[unsafe_offset=psrc])


def patchify[
    target: StaticString, BATCH: Int, C: Int, IMG: Int, PATCH_D: Int
](
    ctx: Optional[DeviceContext],
    img: Pointer[Scalar[DT], MutAnyOrigin],
    dst_buf: Pointer[Scalar[DT], MutAnyOrigin],
) raises:
    comptime GRID = IMG // PATCH_D
    comptime PATCH_PX = C * PATCH_D * PATCH_D
    comptime NQ = GRID * GRID
    comptime IMGN = C * IMG * IMG
    comptime OUTN = NQ * PATCH_PX
    comptime if target == "cpu":
        for b in range(BATCH):
            for p in range(NQ):
                var ph = p // GRID
                var pw = p % GRID
                for c in range(C):
                    for i in range(PATCH_D):
                        for j in range(PATCH_D):
                            var dst = (
                                b * OUTN
                                + p * PATCH_PX
                                + c * (PATCH_D * PATCH_D)
                                + i * PATCH_D
                                + j
                            )
                            var src = (
                                b * IMGN
                                + c * (IMG * IMG)
                                + (ph * PATCH_D + i) * IMG
                                + (pw * PATCH_D + j)
                            )
                            dst_buf[unsafe_offset=dst] = img[unsafe_offset=src]
    else:
        var c = ctx.value()
        var img_lt = tile[BATCH, IMGN](img)
        var out_lt = tile[BATCH, OUTN](dst_buf)
        comptime total = BATCH * OUTN
        comptime n_blocks = (total + TPB - 1) // TPB
        c.enqueue_function[_patchify_kernel[BATCH, C, IMG, PATCH_D]](
            img_lt, out_lt, grid_dim=n_blocks, block_dim=TPB
        )


def unpatchify[
    target: StaticString, BATCH: Int, C: Int, IMG: Int, PATCH_D: Int
](
    ctx: Optional[DeviceContext],
    patches: Pointer[Scalar[DT], MutAnyOrigin],
    img: Pointer[Scalar[DT], MutAnyOrigin],
) raises:
    comptime GRID = IMG // PATCH_D
    comptime PATCH_PX = C * PATCH_D * PATCH_D
    comptime NQ = GRID * GRID
    comptime IMGN = C * IMG * IMG
    comptime OUTN = NQ * PATCH_PX
    comptime if target == "cpu":
        for b in range(BATCH):
            for c in range(C):
                for y in range(IMG):
                    for x in range(IMG):
                        var ph = y // PATCH_D
                        var i = y % PATCH_D
                        var pw = x // PATCH_D
                        var j = x % PATCH_D
                        var p = ph * GRID + pw
                        var psrc = (
                            b * OUTN
                            + p * PATCH_PX
                            + c * (PATCH_D * PATCH_D)
                            + i * PATCH_D
                            + j
                        )
                        var dst = b * IMGN + c * (IMG * IMG) + y * IMG + x
                        img[unsafe_offset=dst] = patches[unsafe_offset=psrc]
    else:
        var cc = ctx.value()
        var p_lt = tile[BATCH, OUTN](patches)
        var img_lt = tile[BATCH, IMGN](img)
        comptime total = BATCH * IMGN
        comptime n_blocks = (total + TPB - 1) // TPB
        cc.enqueue_function[_unpatchify_kernel[BATCH, C, IMG, PATCH_D]](
            p_lt, img_lt, grid_dim=n_blocks, block_dim=TPB
        )
