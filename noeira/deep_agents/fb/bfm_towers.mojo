"""BFM-Zero's network towers as `Module` compositions — G3.2.

`docs/BFM_ZERO_G1_REPRODUCTION.md` §12, from the reference's
`humanoidverse/agents/nn_models.py` (the `residual` model family the
released `config.json` selects for F, the critics and the actor; the plain
MLPs for B and the discriminator). Every alias below is a composition of
existing, individually gated primitives — nothing here has a forward or a
vjp of its own — so the gate (`tests/nn/test_bfm_towers_vs_torch.mojo`)
checks the COMPOSITION against the reference's classes run in torch on the
same weights: the slicing of the flat `[s | a | z]` row the FB trainer
hands every net, the two-embedding concat order, the residual wiring and
the block layout.

    Block(in, out)      = LayerNorm(in) → Linear(in, out) → Mish      (`Block`, activation=True)
    BlockLinear(in,out) = LayerNorm(in) → Linear(in, out)             (`Block`, activation=False)
    ResBlock(h)         = x + Mish(Linear(LayerNorm(x)))              (`ResidualBlock`)
    Embed(in, h, L)     = Block(in, h) → ResBlock(h) × (L − 2) → Block(h, h/2)   (`residual_embedding`)

    FTower   [s|a|z] → [ Embed_sa([s|a]) | Embed_z([s|z]) ] → ResBlock × L → BlockLinear(h, out)
             (`ResidualForwardMap`: embeddings use `hidden_layers` = L; concat order sa, z)
    ActorTower [s|z] → [ Embed_s(s) | Embed_z([s|z]) ] → ResBlock × L → BlockLinear(h, act) → tanh
             (`ResidualActor`: embeddings use `embedding_layers` = 2; concat order s, z)
    BNet     s → Linear(obs, hb) → LayerNorm → Tanh → Linear(hb, d) → Norm
             (`BackwardMap`, `hidden_layers 1`, `norm True`: Norm = sqrt(d)·x/‖x‖, no params)
    DNet     [s|z] → Linear(obs+d, hd) → LayerNorm → Tanh → [Linear → ReLU] × 2 → Linear(hd, 1)
             (`Discriminator`, `hidden_layers 3`)

⚠ THE FLAT ROW. `FBTrainer` packs one `[s | a | z]` row per sample
(`_A_OFF = OBS`, `_Z_OFF = OBS + ACT`), the actor and D see `[s | z]`. The
reference concatenates `[obs, z]` for the z-embedding and `[obs, action]`
for the sa-embedding; `Slice` and a two-way `Parallel[Slice, Slice]` rebuild
exactly those inputs from the flat row. The discriminator's own input in
the reference is `cat([z, obs])` — z FIRST — which is a permutation of the
first Linear's columns and changes nothing for a net trained here; our D
takes `[s | z]` like the walker's, and the gate does not cover D.

⚠ `Norm()` IS RMSNORM WITH γ FROZEN AT ONE. `sqrt(d) · x / ‖x‖` equals
`RMSNorm` with unit gain and a vanishing epsilon (1e-12, as `F.normalize`'s); `RMSNorm` carries a
trainable γ, so it is wrapped in `StopGradParams`, which keeps γ at its
init of 1 while letting the gradient flow to the Linear below.

⚠ UNFUSED `Linear` → `Mish`, NOT `LinearMish`. The fused `LinearAct[.., MishOp]`
matches torch forward but its CPU backward is wrong by ~0.2 relative on
both the input and the parameter gradients (measured by the gate's fused
probe, 2026-09-09: Linear+Mish 0.20 / 0.25); the unfused pair is right to
1e-7. Recorded in `docs/BFM_ZERO_G1_REPRODUCTION.md` §12.3 as an open nn
defect; the towers do not depend on it.

⚠ `Repeat[N, …]` NEEDS N ≥ 1. The actor's embeddings have `embedding_layers
2`, i.e. no residual block between the two `Block`s, so they use `Embed2`
rather than `Embed[…, 2]`.

Sizes: the release is h 2048, L 6 (440 M); Fig. 13's 60 M point is h 1024,
L 3, which G3 uses. Parameters of `FTower[527, 29, 256, 1024, 3, 256]` are
~7.5 M per head; the walker-era `FBTrainer` instantiates F twice.
"""

from noeira.nn.constants import DT
from noeira.nn.combinators.parallel import Parallel
from noeira.nn.combinators.repeat import Repeat
from noeira.nn.combinators.residual import Residual
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.stop_grad_params import StopGradParams
from noeira.nn.primitives.activations import Mish, ReLU, Tanh
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.primitives.slice import Slice


comptime BFMBlock[IN: Int, OUT: Int] = Sequential[LayerNorm[IN], Linear[IN, OUT], Mish[OUT]]
comptime BFMBlockLinear[IN: Int, OUT: Int] = Sequential[LayerNorm[IN], Linear[IN, OUT]]
comptime BFMResBlock[H: Int] = Residual[Sequential[LayerNorm[H], Linear[H, H], Mish[H]]]

# `residual_embedding(in, h, L)`, L >= 3
comptime BFMEmbed[IN: Int, H: Int, L: Int] = Sequential[
    BFMBlock[IN, H], Repeat[L - 2, BFMResBlock[H]], BFMBlock[H, H // 2]
]
# `residual_embedding(in, h, 2)`: no residual block between the two Blocks
comptime BFMEmbed2[IN: Int, H: Int] = Sequential[BFMBlock[IN, H], BFMBlock[H, H // 2]]

# `ResidualForwardMap` on the flat `[s | a | z]` row (one head; F is twinned
# by the trainer). Embeddings use L, as the reference's `hidden_layers`.
comptime BFMFTower[OBS: Int, ACT: Int, D: Int, H: Int, L: Int, OUT: Int] = Sequential[
    Parallel[
        Sequential[Slice[OBS + ACT + D, 0, OBS + ACT], BFMEmbed[OBS + ACT, H, L]],
        Sequential[
            Parallel[Slice[OBS + ACT + D, 0, OBS], Slice[OBS + ACT + D, OBS + ACT, OBS + ACT + D]],
            BFMEmbed[OBS + D, H, L],
        ],
    ],
    Repeat[L, BFMResBlock[H]],
    BFMBlockLinear[H, OUT],
]

# `ResidualActor` on the flat `[s | z]` row: embeddings of depth 2, tanh mean.
comptime BFMActorTower[OBS: Int, D: Int, H: Int, L: Int, ACT: Int] = Sequential[
    Parallel[
        Sequential[Slice[OBS + D, 0, OBS], BFMEmbed2[OBS, H]],
        BFMEmbed2[OBS + D, H],
    ],
    Repeat[L, BFMResBlock[H]],
    BFMBlockLinear[H, ACT],
    Tanh[ACT],
]

# ── the per-net observation FILTERS (docs §12.34-12.35) ───────────────
#
# `released/new_model/config.json` gives each net its own key list. With the
# packed row laid out
#
#     [ state 64 | privileged 463 | last_action 29 | history 372 | z 256 ]
#       \________ SD + PRIV = 527 ________/ \_____ EX = 401 _____/
#
# the four filters are:
#
#     f, critic     everything            -> the row as it is
#     b             [0, 527)              -> BFMBNetFiltered
#     discriminator [0, 527) ++ z         -> BFMDNetFiltered
#     actor         [0, 64) ++ [527, +z)  -> BFMActorTowerFiltered
#
# `b` and `discriminator` take the 527 PREFIX, which is why the privileged
# block sits in the middle rather than at the end: their filters are exactly
# our pre-existing observation, so their weights and their inputs are
# unchanged by the extension.
#
# ⚠ `discriminator` needs `z` too, and `z` is at the END of the packed row, so
# its filter is a two-slice concat like the actor's — NOT a prefix.

# `BackwardMap` on `state + privileged_state` only.
comptime BFMBNetFiltered[OBSF: Int, SP: Int, D: Int, HB: Int] = Sequential[
    Slice[OBSF, 0, SP], BFMBNet[SP, D, HB]
]

# `Discriminator` on `[state + privileged_state | z]`.
comptime BFMDNetFiltered[OBSF: Int, SP: Int, D: Int, HD: Int] = Sequential[
    Parallel[Slice[OBSF + D, 0, SP], Slice[OBSF + D, OBSF, OBSF + D]],
    BFMDNet[SP, D, HD],
]

# ── the actor's VIEW of the packed row (docs §12.34) ──────────────────
#
# `released/new_model/config.json` filters the observation dict per net, and
# the ACTOR's keys are `state, last_action, history_actor` — it does NOT see
# `privileged_state`, which `f`, `critic` and `aux_critic` do. Our packed row
# is
#
#     [ state 64 | privileged 463 | last_action 29 | history 372 | z 256 ]
#          SD          OBSF-SD-EX         EX = 401                   D
#
# so the actor's 721 is the row MINUS the privileged block — two slices
# concatenated, `[0, SD)` and `[OBSF - EX, OBSF + D)`. The privileged block
# stays in the MIDDLE so `b` and `discriminator` keep reading `[0, 527)`
# contiguously, which is exactly `state + privileged_state`, their own keys.
#
# ⚠ Feeding the actor the privileged block is not a free extra: §12.34
# measured our privileged-actor policy at 1.32-1.83x the RELEASED actor on the
# same clips in an environment G2 proved identical. Matching the reference
# means giving those 463 dims UP.
comptime BFMActorView[OBSF: Int, SD: Int, EX: Int, D: Int] = Parallel[
    Slice[OBSF + D, 0, SD],
    Slice[OBSF + D, OBSF - EX, OBSF + D],
]

# `ResidualActor` on the reference's OWN actor keys: `state | last_action |
# history | z`, with `embedding_layers 2` embeddings (`BFMEmbed2`) and an
# `hidden_layers`-deep residual trunk — `nn_models.py:523-527`.
comptime BFMActorTowerFiltered[
    OBSF: Int, SD: Int, EX: Int, D: Int, H: Int, L: Int, ACT: Int
] = Sequential[
    BFMActorView[OBSF, SD, EX, D],
    BFMActorTower[SD + EX, D, H, L, ACT],
]

# `BackwardMap` with `hidden_layers 1`, `norm True`
comptime BFMBNet[OBS: Int, D: Int, HB: Int] = Sequential[
    Linear[OBS, HB], LayerNorm[HB], Tanh[HB], Linear[HB, D],
    StopGradParams[RMSNorm[D, Scalar[DT](1e-12)]]
]

# `Discriminator` with `hidden_layers 3`, on `[s | z]`
comptime BFMDNet[OBS: Int, D: Int, HD: Int] = Sequential[
    Linear[OBS + D, HD], LayerNorm[HD], Tanh[HD],
    Linear[HD, HD], ReLU[HD],
    Linear[HD, HD], ReLU[HD],
    Linear[HD, 1],
]
