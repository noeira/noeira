# +--------------------------------------------------------------------------+ #
# | ACTLossGraph — the whole CVAE objective as one ComputeGraph
# +--------------------------------------------------------------------------+ #
"""`DETRVAE.forward` + `ACTPolicy.__call__`'s loss, as one graph.

Mirrored the first LeWM port's loss graph (removed 2026-10-03): every component is an owned node,
so a single Adam iterating `graph.for_each_param` trains the whole model.

## Inputs

    qpos        (B, QPOS)                 normalized joint positions
    images      (B, N_CAM*3*IMG_H*IMG_W)  /255 then ImageNet-normalized, CHW
    actions     (B, K*ADIM)               normalized, padded past episode end
    enc_valid   (B, K+2)                  1.0 = attend, 0.0 = padding

⚠ `enc_valid` carries the CVAE encoder's mask for `[CLS] | qpos | a_1..a_K`, so
its first two entries are always 1.0 (`detr_vae.py:96` prepends
`cls_joint_is_pad = False, False`). The chunk's own mask is `Slice[2:]` of it —
one input rather than two, so the two can never disagree.

## Structure

    CVAE encoder     Concat([CLS], Lin(qpos), Tok Lin(actions))     (K+2 tokens)
                       -> N_ENC x DETREncoderLayerMasked(c = [pos1d | valid])
                       -> Slice token 0 -> Linear[DIM, 2*LATENT] =: latent_info
                       -> reparameterize -> Scale["zs"] -> Linear[LATENT, DIM]

    vision           Tok[N_CAM] ResNet18 -> Tok[N_CAM] Transpose2D
                       -> Tok[NTOK] Linear[512, DIM]                =: src
    transformer      Concat(latent_tok, Lin(qpos), src)             (2+NTOK)
                       -> N_ENC x DETREncoderLayer(c = mem_pos)     =: memory
                       mem_pos = Concat(learned 2 tokens, 2-D sine x N_CAM)

    decoder          LearnedQueries (content) -> N_DEC x DETRDecoderLayer(
                                     c = [query_embed | memory+pos | memory])
                       -> Tok LayerNorm -> Tok Linear[DIM, ADIM]    =: a_hat

    loss             L1MaskedPerSample(a_hat, actions, valid)
                       + Scale["kls"](KL(latent_info))
                       + Scale["l1ds"](L1MaskedPerSample(Δa_hat, Δactions, valid[1:]))
                     Δ = positions 1..K-1 minus 0..K-2 (Slice, Scale(-1), Add);
                     the shape term, weight ACT_SHAPE_WEIGHT (0 = the paper)

## ⚠ ONE GRAPH SERVES TRAINING AND INFERENCE

The reference skips the CVAE encoder at test time and sets `z = 0`
(`detr_vae.py:110`). Here `set_node_attr["zs", "multiplier"](0.0)` makes the
latent token `latent_out_proj(0)` — its bias alone — which is the same number by
construction, not an approximation. The cost is one wasted CVAE-encoder pass
(K+2 tokens, small beside the ResNet). This avoids a second graph and a
parameter-sync path between them, which is the more likely source of a silent
divergence. `actions` must still be fed at inference; zeros are fine, since
nothing downstream of the scaled-to-zero latent reads them.

## ⚠ `DEC_LAYERS` defaults to 1 and that is not a simplification

`detr_vae.py:139` reads `self.transformer(...)[0]`, and `build_transformer`
passes `return_intermediate_dec=True`, so the decoder returns a LAYER-indexed
stack and `[0]` selects the FIRST layer's output. Layers 2..7 of the published
model receive no gradient and cannot affect the prediction — `dec_layers=7` is
output-equivalent to `dec_layers=1`. `RepeatConditional` returns the LAST
layer's output, so at `DEC_LAYERS=1` the two coincide exactly. Set
`ACT_DEC_LAYERS = 7` to get LeRobot's corrected reading (last layer, all layers
trained); it is a different model, and a better one, but not this paper's.

## Deviations, all deliberate

* **`is_pad_head` is omitted.** `detr_vae.py:57` declares
  `nn.Linear(hidden_dim, 1)` and `forward` returns `is_pad_hat`, which
  `policy.py` computes and never puts in the loss. Dead weight with a dead
  gradient.
* **BatchNorm sees `B*N_CAM` rows, not `B`.** `Tokenwise[N_CAM, ResNet18]` runs
  the shared backbone over the cameras as one flattened batch; the reference
  calls it once per camera. Identical in eval (running statistics), slightly
  different batch statistics while training. Sharing ONE backbone across cameras
  IS the reference (`self.backbones[0]  # HARDCODED`).
"""

from noeira.nn.core.module import Module
from noeira.nn import (
    Add,
    ComputeGraph,
    Concat,
    InputSlot,
    LayerNorm,
    LearnedQueries,
    Linear,
    Node,
    RepeatConditional,
    Scale,
    Slice,
    Tokenwise,
    Transpose2D,
)
from noeira.nn.models.resnet18 import (
    RESNET18_OUT_CH,
    ResNet18Backbone,
    ResNet18OutH,
    ResNet18OutW,
)
from noeira.nn.primitives.gaussian_vae import (
    GaussianKLStdNormal,
    GaussianReparam,
)
from noeira.nn.primitives.l1_masked_per_sample import L1MaskedPerSample
from noeira.nn.primitives.sinusoidal_pos_tokens import (
    SinusoidalPos2DTokens,
    SinusoidalPos1DTokens,
)

from .layers import DETRDecoderLayer, DETREncoderLayer, DETREncoderLayerMasked


comptime ACTLossGraph[
    QPOS: Int,
    ADIM: Int,
    N_CAM: Int,
    IMG_H: Int,
    IMG_W: Int,
    K: Int,
    DIM: Int,
    HEADS: Int,
    FF: Int,
    LATENT: Int,
    N_ENC: Int,
    N_DEC: Int,
    P: Float64 = 0.1,
    # Derived — spelled as defaulted parameters because Mojo comptime aliases
    # cannot introduce local bindings, and repeating these expressions inline
    # (they appear 20+ times below) is where a transcription error would hide.
    # ⚠ `BACKBONE` is a parameter so a GPU-vs-CPU gate can swap ResNet18 for a
    # two-conv stub. That gate instantiates the WHOLE graph twice (once per
    # target), and ResNet18 is 20 Conv2D + 20 BatchNorm2D — 80 kernel
    # instantiations on its own, which is what made the gate untenable to
    # build. The vision tower's own GPU path is gated separately and cheaply
    # (`test_resnet18_gpu.mojo`); the model gate needs a backbone, not THIS
    # backbone. Default is unchanged, so every existing caller is identical.
    FEAT_CH: Int = RESNET18_OUT_CH,
    OH: Int = ResNet18OutH[IMG_H],
    OW: Int = ResNet18OutW[IMG_W],
    NTOK: Int = N_CAM * ResNet18OutH[IMG_H] * ResNet18OutW[IMG_W],
    MEM: Int = 2 + N_CAM * ResNet18OutH[IMG_H] * ResNet18OutW[IMG_W],
    ENC_SEQ: Int = K + 2,
    BACKBONE: Module = ResNet18Backbone[3, IMG_H, IMG_W],
] = ComputeGraph[
    InputSlot["qpos", QPOS],
    InputSlot["images", N_CAM * 3 * IMG_H * IMG_W],
    InputSlot["actions", K * ADIM],
    InputSlot["enc_valid", ENC_SEQ],
    # ── CVAE encoder ─────────────────────────────────────────────────────
    Node["cls", LearnedQueries[QPOS, 1, DIM], "qpos"],
    Node["qenc", Linear[QPOS, DIM], "qpos"],  # encoder_joint_proj
    Node["aenc", Tokenwise[K, Linear[ADIM, DIM]], "actions"],  # ..._action_proj
    Node["einp", Concat[DIM, DIM, K * DIM], "cls", "qenc", "aenc"],
    Node["epos", SinusoidalPos1DTokens[QPOS, ENC_SEQ, DIM], "qpos"],
    Node["ec", Concat[ENC_SEQ * DIM, ENC_SEQ], "epos", "enc_valid"],
    Node[
        "cvae",
        RepeatConditional[
            N_ENC, DETREncoderLayerMasked[DIM, HEADS, ENC_SEQ, FF, P]
        ],
        "einp",
        "ec",
    ],
    Node["clsout", Slice[ENC_SEQ * DIM, 0, DIM], "cvae"],
    Node["latinfo", Linear[DIM, 2 * LATENT], "clsout"],  # latent_proj
    Node["z", GaussianReparam[LATENT], "latinfo"],
    Node["zs", Scale[LATENT], "z"],  # 1.0 train / 0.0 eval
    Node["lattok", Linear[LATENT, DIM], "zs"],  # latent_out_proj
    # ── vision ───────────────────────────────────────────────────────────
    Node["feat", Tokenwise[N_CAM, BACKBONE], "images"],
    Node["featt", Tokenwise[N_CAM, Transpose2D[FEAT_CH, OH * OW]], "feat"],
    Node["src", Tokenwise[NTOK, Linear[FEAT_CH, DIM]], "featt"],
    Node["prop", Linear[QPOS, DIM], "qpos"],  # input_proj_robot_state
    Node["meminp", Concat[DIM, DIM, NTOK * DIM], "lattok", "prop", "src"],
    # ── transformer encoder ──────────────────────────────────────────────
    Node["addpos", LearnedQueries[QPOS, 2, DIM], "qpos"],  # additional_pos_embed
    Node["impos", SinusoidalPos2DTokens[QPOS, DIM, OH, OW, N_CAM], "qpos"],
    Node["mempos", Concat[2 * DIM, NTOK * DIM], "addpos", "impos"],
    Node[
        "memory",
        RepeatConditional[N_ENC, DETREncoderLayer[DIM, HEADS, MEM, FF, P]],
        "meminp",
        "mempos",
    ],
    # ── transformer decoder ──────────────────────────────────────────────
    Node["kmem", Add[MEM * DIM], "memory", "mempos"],  # layer-invariant
    Node["qpe", LearnedQueries[QPOS, K, DIM], "qpos"],  # query_embed
    # ⚠ THE DECODER'S TARGET TOKENS ARE A SECOND LEARNED EMBEDDING, NOT THE
    # REFERENCE'S ZEROS (`transformer.py:72`). With `tgt = 0` the only route
    # from "which chunk position" to the output is the cross-attention
    # WEIGHTS over the memory (self-attention values are zero, the residual
    # stream is zero, the FFN sees the attended memory alone), and on LIBERO
    # every fit — per-task, multi-task, 300k steps, unit-scale queries, KL
    # 0.1, a shape term — converged to ONE action for all 40 positions
    # (`libero_act_inspect`: decoder spread 0.0003-0.0009 at rms 1, from
    # 0.0157 fresh): the per-position part of the output is a small residual
    # the L1 fit removes while it fits the per-state mean, and flat is the
    # attractor. A learned content query per position (the construction
    # conditional-DETR-style decoders use) puts the position in the residual
    # stream: self-attention values, the residual and the FFN all see it, a
    # per-position offset per state is trivially expressible, and flat is no
    # longer the easy solution. `qpe` stays the positional query added to q
    # and k as before. A pre-change checkpoint does not load (one more Param).
    Node["tgt0", LearnedQueries[QPOS, K, DIM], "qpos"],  # content queries
    Node[
        "dc", Concat[K * DIM, MEM * DIM, MEM * DIM], "qpe", "kmem", "memory"
    ],
    Node[
        "hs",
        RepeatConditional[
            N_DEC, DETRDecoderLayer[DIM, HEADS, K, MEM, FF, P]
        ],
        "tgt0",
        "dc",
    ],
    Node["hsn", Tokenwise[K, LayerNorm[DIM]], "hs"],  # decoder_norm
    Node["ahat", Tokenwise[K, Linear[DIM, ADIM]], "hsn"],  # action_head
    # ── loss ─────────────────────────────────────────────────────────────
    Node["valid", Slice[ENC_SEQ, 2, ENC_SEQ], "enc_valid"],
    Node["l1", L1MaskedPerSample[K, ADIM], "ahat", "actions", "valid"],
    Node["kl", GaussianKLStdNormal[LATENT], "latinfo"],
    Node["kls", Scale[1], "kl"],  # kl_weight
    Node["lossa", Add[1], "l1", "kls"],
    # ── the chunk-shape term: first differences along the chunk ─────────
    # `*_lo_neg` carry multiplier -1 (set in `ACTTrainer.make`); `l1ds` the
    # shape weight. valid[1:] because a difference at t needs t and t+1 real,
    # and `valid` is a prefix mask.
    Node["ahat_hi", Slice[K * ADIM, ADIM, K * ADIM], "ahat"],
    Node["ahat_lo", Slice[K * ADIM, 0, (K - 1) * ADIM], "ahat"],
    Node["ahat_lo_neg", Scale[(K - 1) * ADIM], "ahat_lo"],
    Node["dpred", Add[(K - 1) * ADIM], "ahat_hi", "ahat_lo_neg"],
    Node["act_hi", Slice[K * ADIM, ADIM, K * ADIM], "actions"],
    Node["act_lo", Slice[K * ADIM, 0, (K - 1) * ADIM], "actions"],
    Node["act_lo_neg", Scale[(K - 1) * ADIM], "act_lo"],
    Node["dtgt", Add[(K - 1) * ADIM], "act_hi", "act_lo_neg"],
    Node["valid_hi", Slice[K, 1, K], "valid"],
    Node["l1d", L1MaskedPerSample[K - 1, ADIM], "dpred", "dtgt", "valid_hi"],
    Node["l1ds", Scale[1], "l1d"],  # shape_weight
    Node["loss", Add[1], "lossa", "l1ds"],
]
