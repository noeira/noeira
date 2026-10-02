"""LeWM nn nets — encoder, action embedder, AR predictor, pred projector.

Pure `Sequential` aliases over existing nn composites + the Phase A/B
primitives. No new compute. Mirrors the legacy `experimental/lewm/`
architecture (docs/LEWM_PORT_PLAN.md §3) on nn.

- `LeWMEncoder`  : ViT body (PatchEmbed → pos BiasAdd → N×TransformerBlock →
                   per-token LayerNorm → TokenMean) + JEPA projector MLP
                   (Linear → BatchNorm1D → GELU → Linear). Maps a flattened
                   image (B, IN_CH·IMG·IMG) → (B, EMB). Run at BATCH=B·T.
- `ActionEmbedder`: per-step Conv1d-k1 stack (Tokenwise LinearSwish×2 + Linear)
                   mapping (B, T·ACT) → (B, T·EMB).
- `ARPredictor`  : RepeatConditional[DEPTH, ConditionalTransformerBlock] over
                   the H-token context (the learnable position BiasAdd is a
                   separate node in the JEPA graph, applied to x before this —
                   it can't chain into an ARITY=2 module).
- `PredProj`     : per-token projector on the predictor output (Tokenwise).
"""

from noeira.nn import (
    Sequential,
    Repeat,
    RepeatConditional,
    Linear,
    LinearSwish,
    BatchNorm1D,
    GELUTanh,
    LayerNorm,
    TokenMean,
    BiasAdd,
    ConditionalTransformerBlock,
    LearnedTokens,
    Slice,
    Tokenwise,
)
from noeira.nn.models.vit import PatchEmbed
from noeira.nn.models.transformer import TransformerBlock


# Projector MLP: (B, HIDDEN) → (B, EMB). BatchNorm1D matches the reference.
comptime LeWMProjector[HIDDEN: Int, PROJ_H: Int, EMB: Int] = Sequential[
    Linear[HIDDEN, PROJ_H],
    BatchNorm1D[PROJ_H],
    GELUTanh[PROJ_H],
    Linear[PROJ_H, EMB],
]


# Encoder: image → embedding. Run at effective BATCH = B·T (one image per row).
comptime LeWMEncoder[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    N_PATCHES: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
    FF_MULT: Int = 4,
] = Sequential[
    PatchEmbed[IN_CH, IMG, IMG, PATCH, HIDDEN, N_PATCHES],
    BiasAdd[N_PATCHES * HIDDEN],
    Repeat[
        ENC_LAYERS,
        TransformerBlock[HIDDEN, ENC_HEADS, N_PATCHES, FF_MULT * HIDDEN, False],
    ],
    Tokenwise[N_PATCHES, LayerNorm[HIDDEN]],
    TokenMean[N_PATCHES, HIDDEN],
    LeWMProjector[HIDDEN, PROJ_H, EMB],
]


# CLS-token encoder variant: prepend a learnable [CLS] token, run the
# transformer over N_PATCHES+1 tokens, then read the CLS token (slice [0:HID])
# instead of mean-pooling. Same external interface as `LeWMEncoder` (image →
# (B, EMB)), so it drops into the loss graph identically. Motivation: the
# closed-loop probe showed the mean-pooled latent under-encodes the small
# agent/pusher (washed out across N_PATCHES); a CLS token can attend
# selectively to control-relevant patches (the prediction objective rewards
# encoding the pusher, since actions move it). No new primitives — reuses
# `LearnedTokens[…, PREPEND=True]` (prepend CLS) + `Slice` (extract token 0).
# The CLS token is initialized small (std 0.02, ViT convention) via
# LearnedTokens' INIT_STD: fan_in=1 Kaiming would give std≈1.4, a constant
# that swamps the per-image attention signal and collapses the readout (all
# images → near-identical CLS → batch-variance collapse → BatchNorm/SIGReg
# blow up). Mean-pooling is immune; the CLS readout is not.
comptime LeWMEncoderCLS[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    N_PATCHES: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
    FF_MULT: Int = 4,
] = Sequential[
    PatchEmbed[IN_CH, IMG, IMG, PATCH, HIDDEN, N_PATCHES],
    LearnedTokens[N_PATCHES, 1, HIDDEN, True, 0.02],  # prepend [CLS] (ViT init)
    BiasAdd[(N_PATCHES + 1) * HIDDEN],  # pos embed incl CLS
    Repeat[
        ENC_LAYERS,
        TransformerBlock[
            HIDDEN, ENC_HEADS, N_PATCHES + 1, FF_MULT * HIDDEN, False
        ],
    ],
    Tokenwise[N_PATCHES + 1, LayerNorm[HIDDEN]],
    Slice[(N_PATCHES + 1) * HIDDEN, 0, HIDDEN],  # take [CLS] (token 0)
    LeWMProjector[HIDDEN, PROJ_H, EMB],
]


# Action embedder: (B, T·ACT) → (B, T·EMB). Conv1d-k1 ≡ per-token Linear.
#   LinearSwish (Linear+SiLU) ×2 then a bare Linear, all Tokenwise over T.
comptime ActionEmbedder[
    T: Int, ACT: Int, SMOOTHED: Int, EMB: Int, MLP_SCALE: Int = 4
] = Sequential[
    Tokenwise[T, LinearSwish[ACT, SMOOTHED]],
    Tokenwise[T, LinearSwish[SMOOTHED, MLP_SCALE * EMB]],
    Tokenwise[T, Linear[MLP_SCALE * EMB, EMB]],
]


# AR predictor: learnable position embedding over the H-token context, then a
# DEPTH-stack of AdaLN-zero conditional blocks. ARITY=2: forward(x, c) where x
# is the context embeddings (B, H·EMB) and c the action conditioning.
comptime ARPredictor[
    EMB: Int, HEADS: Int, H: Int, FF: Int, DEPTH: Int, HEAD_DIM: Int = 0
] = RepeatConditional[
    DEPTH, ConditionalTransformerBlock[EMB, HEADS, H, FF, HEAD_DIM]
]


# Predictor output projector: per-token MLP (B, H·EMB) → (B, H·EMB).
comptime PredProj[H: Int, EMB: Int, PROJ_H: Int] = Tokenwise[
    H,
    Sequential[
        Linear[EMB, PROJ_H],
        BatchNorm1D[PROJ_H],
        GELUTanh[PROJ_H],
        Linear[PROJ_H, EMB],
    ],
]
