"""Vision Transformer (ViT) composition for nn.storage.

Storage-surface port of `nn/models/vit.mojo`. Compositional aliases over the
shared transformer pieces in `models.transformer` + storage leaves. No structs,
no kernels (imports are already the storage primitives/combinators).
  - `PatchEmbed[in_channels, img_h, img_w, patch_size, embed_dim, n_patches]`
  - `ViT[...]`  Vision Transformer encoder + classification head (non-causal)
"""

from ..primitives.conv2d import Conv2D
from ..primitives.linear import Linear
from ..primitives.layer_norm import LayerNorm
from ..primitives.bias_add import BiasAdd
from ..primitives.transpose_2d import Transpose2D
from ..primitives.token_mean import TokenMean
from ..combinators.sequential import Sequential
from ..combinators.repeat import Repeat
from ..combinators.tokenwise import Tokenwise
from .transformer import TransformerBlockFused


# PatchEmbed: image → patch tokens.
#   Conv2D (channel-major patches) → Transpose2D (→ patch-major).
#   (BATCH, in_channels*img_h*img_w) → (BATCH, n_patches*embed_dim).
comptime PatchEmbed[
    in_channels: Int,
    img_h: Int,
    img_w: Int,
    patch_size: Int,
    embed_dim: Int,
    n_patches: Int,
] = Sequential[
    Conv2D[in_channels, embed_dim, patch_size, patch_size, 0, img_h, img_w],
    Transpose2D[embed_dim, n_patches],
]


# ViT: Vision Transformer encoder + classification head (non-causal).
#   PatchEmbed → position BiasAdd → N×TransformerBlockFused → LayerNorm →
#   TokenMean (mean-pool patches) → Linear head.
comptime ViT[
    in_channels: Int,
    img_h: Int,
    img_w: Int,
    patch_size: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    n_patches: Int,
    n_classes: Int,
    ff_mult: Int = 4,
    use_max: Bool = True,
] = Sequential[
    PatchEmbed[in_channels, img_h, img_w, patch_size, embed_dim, n_patches],
    BiasAdd[n_patches * embed_dim],
    Repeat[
        n_layers,
        TransformerBlockFused[
            embed_dim, n_heads, n_patches, ff_mult * embed_dim, False, use_max
        ],
    ],
    Tokenwise[n_patches, LayerNorm[embed_dim]],
    TokenMean[n_patches, embed_dim],
    Linear[embed_dim, n_classes],
]
