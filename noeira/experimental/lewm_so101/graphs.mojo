"""The SO-101 world models' training graphs: `LeWMLossGraph` with
proprioception in the latent (v2) and with an inverse-dynamics term (IDM).

noeira-docs/SO101_LEWM_PLAN.md S1 and its 24 Oct section. Closed 2 Nov: a
world model trained in sim did not predict the real arm. Kept for the record
and for the examples in `examples/lewm/lewm_so101_*`.
"""

from noeira.nn.constants import DT
from noeira.nn import (
    ComputeGraph,
    InputSlot,
    Node,
    Tokenwise,
    Slice,
    BiasAdd,
    RepeatConditional,
    LayerNorm,
    MSEPerSample,
    SIGReg,
    Scale,
    Add,
    Linear,
    GELU,
)
from noeira.nn.primitives.grad_gate import GradGate
from noeira.deep_agents.lewm.model import (
    LeWMEncoder,
    LeWMActionEmbedder,
    LeWMPredictorBlock,
    LeWMProjector,
    TORCH_LN_EPS,
)


comptime LeWMLossGraphProprio[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
    T: Int,
    ACT: Int,
    PROP: Int,
    H: Int,
    N_PREDS: Int,
    PRED_HEADS: Int,
    PRED_DIM_HEAD: Int,
    PRED_FF: Int,
    DEPTH: Int,
    SIG_PROJ: Int,
    SIG_KNOTS: Int,
] = ComputeGraph[
    InputSlot["pixels", T * IN_CH * IMG * IMG],
    InputSlot["actions", T * ACT],
    InputSlot["proprio", T * PROP],
    Node[
        "emb",
        Tokenwise[
            T,
            LeWMEncoder[
                IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, PROJ_H
            ],
        ],
        "pixels",
    ],
    Node["prop_emb", LeWMActionEmbedder[T, PROP, EMB], "proprio"],
    Node["z", Add[T * EMB], "emb", "prop_emb"],
    Node["act_emb", LeWMActionEmbedder[T, ACT, EMB], "actions"],
    Node["ctx_x", Slice[T * EMB, 0, H * EMB], "z"],
    Node["ctx_a", Slice[T * EMB, 0, H * EMB], "act_emb"],
    Node["tgt", Slice[T * EMB, N_PREDS * EMB, (N_PREDS + H) * EMB], "z"],
    Node["tgt_sg", GradGate[H * EMB], "tgt"],
    Node["x_pe", BiasAdd[H * EMB], "ctx_x"],
    Node[
        "pred_raw",
        RepeatConditional[
            DEPTH,
            LeWMPredictorBlock[
                EMB, PRED_HEADS, H, PRED_FF, PRED_DIM_HEAD
            ],
        ],
        "x_pe",
        "ctx_a",
    ],
    Node["pred_ln", Tokenwise[H, LayerNorm[EMB, DT, TORCH_LN_EPS]], "pred_raw"],
    Node["pred", Tokenwise[H, LeWMProjector[EMB, PROJ_H, EMB]], "pred_ln"],
    Node["pl", MSEPerSample[H * EMB], "pred", "tgt_sg"],
    Node["sig", SIGReg[EMB, T, SIG_PROJ, SIG_KNOTS], "z"],
    Node["sig_s", Scale[1], "sig"],
    Node["loss", Add[1], "pl", "sig_s"],
]
"""`LeWMLossGraph` with PROPRIOCEPTION in the latent (DINO-WM's recipe;
noeira-docs/SO101_LEWM_PLAN.md S1 v2): each frame's `PROP` proprio words go
through their own embedder (`prop_emb`, the action embedder's shape) and are
ADDED to the image latent, z = enc(pixels) + prop_emb(proprio). Everything
downstream — the predictor's context, its targets, SIGReg — reads z, so the
prediction loss scores the next frame's arm state as well as its picture, and
a planner's cost sees it. Node names are the reference's (`emb` is still the
image encoder: `emb.0.*` loads unchanged), plus `prop_emb` and `z`."""


# ── inverse dynamics (H-JEPA's anti-collapse term) ────────────────────────
#
# H-JEPA (arXiv 2610.06805) §4.3 eq. 8: a head I regresses the action between
# two consecutive state latents,
#
#     L_idm = 1/(T−1) Σ_t  1/d_a ‖ I(z_t, z_{t+1}) − sg[a_t] ‖²
#
# weighted γ (100 on DROID, with SIGReg λ 0.08). With one scene per episode a
# purely predictive objective admits a SLOW-FEATURE solution — the encoder
# keeps the scene, which is predictable, and drops the arm, which is not
# without its actions; only a term on consecutive latents rules it out
# (their §L). The SO-101 world models fell into it once each episode was
# rendered under one look (noeira-docs/SO101_LEWM_PLAN.md, 23 Oct).
#
# Here: the action block of frame t is the FS ticks of target change from
# frame t to frame t + 1 (`so101_data`), so the T − 1 pairs regress blocks
# 0 .. T − 2. The head is ours (the paper gives none): an MLP on the pair,
# Linear(2·EMB → IDM_H) written as two Linears on z_t and z_{t+1} summed (the
# same map as a Linear on their concatenation, with one redundant bias),
# exact GELU, Linear(IDM_H → ACT). It reads `emb`, the IMAGE latent, in both
# graphs: in v2 a head on z could read the action off the proprio embedding
# and leave the image encoder collapsed. Its parameters are the `idm_*` nodes;
# every other name is the plain graph's, so a dump loads into the plain
# graph (`LoadDump` reads only the names it is asked for) and the offline
# checks run unchanged.


comptime LeWMLossGraphIDM[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
    T: Int,
    ACT: Int,
    H: Int,
    N_PREDS: Int,
    PRED_HEADS: Int,
    PRED_DIM_HEAD: Int,
    PRED_FF: Int,
    DEPTH: Int,
    SIG_PROJ: Int,
    SIG_KNOTS: Int,
    IDM_H: Int,
] = ComputeGraph[
    InputSlot["pixels", T * IN_CH * IMG * IMG],
    InputSlot["actions", T * ACT],
    Node[
        "emb",
        Tokenwise[
            T,
            LeWMEncoder[
                IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, PROJ_H
            ],
        ],
        "pixels",
    ],
    Node["act_emb", LeWMActionEmbedder[T, ACT, EMB], "actions"],
    Node["ctx_x", Slice[T * EMB, 0, H * EMB], "emb"],
    Node["ctx_a", Slice[T * EMB, 0, H * EMB], "act_emb"],
    Node["tgt", Slice[T * EMB, N_PREDS * EMB, (N_PREDS + H) * EMB], "emb"],
    Node["tgt_sg", GradGate[H * EMB], "tgt"],
    Node["x_pe", BiasAdd[H * EMB], "ctx_x"],
    Node[
        "pred_raw",
        RepeatConditional[
            DEPTH,
            LeWMPredictorBlock[
                EMB, PRED_HEADS, H, PRED_FF, PRED_DIM_HEAD
            ],
        ],
        "x_pe",
        "ctx_a",
    ],
    Node["pred_ln", Tokenwise[H, LayerNorm[EMB, DT, TORCH_LN_EPS]], "pred_raw"],
    Node["pred", Tokenwise[H, LeWMProjector[EMB, PROJ_H, EMB]], "pred_ln"],
    Node["pl", MSEPerSample[H * EMB], "pred", "tgt_sg"],
    Node["sig", SIGReg[EMB, T, SIG_PROJ, SIG_KNOTS], "emb"],
    Node["sig_s", Scale[1], "sig"],
    Node["idm_z0", Slice[T * EMB, 0, (T - 1) * EMB], "emb"],
    Node["idm_z1", Slice[T * EMB, EMB, T * EMB], "emb"],
    Node["idm_h0", Tokenwise[T - 1, Linear[EMB, IDM_H]], "idm_z0"],
    Node["idm_h1", Tokenwise[T - 1, Linear[EMB, IDM_H]], "idm_z1"],
    Node["idm_h", Add[(T - 1) * IDM_H], "idm_h0", "idm_h1"],
    Node["idm_g", GELU[(T - 1) * IDM_H], "idm_h"],
    Node["idm", Tokenwise[T - 1, Linear[IDM_H, ACT]], "idm_g"],
    Node["idm_tgt", Slice[T * ACT, 0, (T - 1) * ACT], "actions"],
    Node["il", MSEPerSample[(T - 1) * ACT], "idm", "idm_tgt"],
    Node["il_s", Scale[1], "il"],
    Node["loss_ps", Add[1], "pl", "sig_s"],
    Node["loss", Add[1], "loss_ps", "il_s"],
]
"""`LeWMLossGraph` (v1) + γ·L_idm on the image latent: loss = pred +
λ·SIGReg + γ·L_idm, per sample; γ = `il_s.multiplier`, the IDM loss itself
is `il` (see the section header)."""


comptime LeWMLossGraphProprioIDM[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
    T: Int,
    ACT: Int,
    PROP: Int,
    H: Int,
    N_PREDS: Int,
    PRED_HEADS: Int,
    PRED_DIM_HEAD: Int,
    PRED_FF: Int,
    DEPTH: Int,
    SIG_PROJ: Int,
    SIG_KNOTS: Int,
    IDM_H: Int,
] = ComputeGraph[
    InputSlot["pixels", T * IN_CH * IMG * IMG],
    InputSlot["actions", T * ACT],
    InputSlot["proprio", T * PROP],
    Node[
        "emb",
        Tokenwise[
            T,
            LeWMEncoder[
                IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, PROJ_H
            ],
        ],
        "pixels",
    ],
    Node["prop_emb", LeWMActionEmbedder[T, PROP, EMB], "proprio"],
    Node["z", Add[T * EMB], "emb", "prop_emb"],
    Node["act_emb", LeWMActionEmbedder[T, ACT, EMB], "actions"],
    Node["ctx_x", Slice[T * EMB, 0, H * EMB], "z"],
    Node["ctx_a", Slice[T * EMB, 0, H * EMB], "act_emb"],
    Node["tgt", Slice[T * EMB, N_PREDS * EMB, (N_PREDS + H) * EMB], "z"],
    Node["tgt_sg", GradGate[H * EMB], "tgt"],
    Node["x_pe", BiasAdd[H * EMB], "ctx_x"],
    Node[
        "pred_raw",
        RepeatConditional[
            DEPTH,
            LeWMPredictorBlock[
                EMB, PRED_HEADS, H, PRED_FF, PRED_DIM_HEAD
            ],
        ],
        "x_pe",
        "ctx_a",
    ],
    Node["pred_ln", Tokenwise[H, LayerNorm[EMB, DT, TORCH_LN_EPS]], "pred_raw"],
    Node["pred", Tokenwise[H, LeWMProjector[EMB, PROJ_H, EMB]], "pred_ln"],
    Node["pl", MSEPerSample[H * EMB], "pred", "tgt_sg"],
    Node["sig", SIGReg[EMB, T, SIG_PROJ, SIG_KNOTS], "z"],
    Node["sig_s", Scale[1], "sig"],
    Node["idm_z0", Slice[T * EMB, 0, (T - 1) * EMB], "emb"],
    Node["idm_z1", Slice[T * EMB, EMB, T * EMB], "emb"],
    Node["idm_h0", Tokenwise[T - 1, Linear[EMB, IDM_H]], "idm_z0"],
    Node["idm_h1", Tokenwise[T - 1, Linear[EMB, IDM_H]], "idm_z1"],
    Node["idm_h", Add[(T - 1) * IDM_H], "idm_h0", "idm_h1"],
    Node["idm_g", GELU[(T - 1) * IDM_H], "idm_h"],
    Node["idm", Tokenwise[T - 1, Linear[IDM_H, ACT]], "idm_g"],
    Node["idm_tgt", Slice[T * ACT, 0, (T - 1) * ACT], "actions"],
    Node["il", MSEPerSample[(T - 1) * ACT], "idm", "idm_tgt"],
    Node["il_s", Scale[1], "il"],
    Node["loss_ps", Add[1], "pl", "sig_s"],
    Node["loss", Add[1], "loss_ps", "il_s"],
]
"""`LeWMLossGraphProprio` (v2) + γ·L_idm on the IMAGE latent `emb`, not on z
(see the section header)."""
