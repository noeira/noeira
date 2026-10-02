"""DreamerV3 networks as composable nn type aliases (redesign).

The encoder, decoder, RSSM prior, and the reward/cont/value head MLPs are
ALL plain `Sequential` chains of existing nn primitives — no hand-written
forward/backward. This is the composable design the spike validated; the
loss heads (twohot / binary / symlog-mse) attach as separate loss ops that
produce the upstream gradient on the logits, after which `Sequential.vjp`
handles the rest.

Activation is `GELU` here (matches the PR4/5b fixtures + reference class
default). The size1m/dmc config uses SiLU — swap `GELU`→`SiLU` (both are
`Elementwise` aliases) when wiring the actual training config.

Layer counts pinned to v1 defaults (enc/dec 2 hidden, prior 2, head MLP 1);
`Repeat[n, Inner]` generalizes the depth later.
"""

from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.init_with import InitWith
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.primitives.elementwise import Elementwise
from noeira.nn.primitives.ops.gelu_tanh_op import GELUTanhOp
from noeira.nn.primitives.symlog import Symlog
from noeira.nn.core.element_op import ElementOp
from noeira.nn.core.initializer import Initializer, Zero, ScaledKaiming


# Inter-layer activation is a comptime op `A` (default `GELUTanhOp`, matching the
# PR4/5b JAX fixtures so the validation spikes stay green). The production
# trainer/agent pass `SwishOp` (size1m/dmc config `act: silu`). `Elementwise
# [DIM, A]` == `GELUTanh[DIM]` when A=GELUTanhOp, `SiLU[DIM]` when A=SwishOp.


# Encoder: symlog(obs) → [Linear, RMSNorm, act] × 2 → tokens[U]
comptime DreamerEncoder[OBS: Int, U: Int, A: ElementOp = GELUTanhOp] = Sequential[
    Symlog[OBS],
    Linear[OBS, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, U], RMSNorm[U], Elementwise[U, A],
]

# Decoder: concat([stoch,deter])[FEATIN] → [Linear,RMSNorm,act]×2 → pred[OBS]
comptime DreamerDecoder[FEATIN: Int, OBS: Int, U: Int, A: ElementOp = GELUTanhOp] = Sequential[
    Linear[FEATIN, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, OBS],
]

# RSSM prior: deter[DETER] → [Linear,RMSNorm,act]×2 → logit[SC]
comptime DreamerPrior[DETER: Int, H: Int, SC: Int, A: ElementOp = GELUTanhOp] = Sequential[
    Linear[DETER, H], RMSNorm[H], Elementwise[H, A],
    Linear[H, H], RMSNorm[H], Elementwise[H, A],
    Linear[H, SC],
]

# Reward head MLP (1 hidden): feat[FEAT] → [Linear,RMSNorm,act] → logits[BINS].
# The OUTPUT Linear's init is declared STRUCTURALLY (InitWith) — paper p.6
# zero-inits the reward-predictor output (reference `rewhead.outscale: 0.0`);
# `OUT_INIT=Kaiming` restores the pre-zero-init optimism (positive-reward
# tasks like CartPole). Replaces the post-hoc `scale_output_graph("rew.3.…")`
# name-path surgery, which failed SILENTLY on refactor.
comptime DreamerRewardMLP[
    FEAT: Int, U: Int, BINS: Int, A: ElementOp = GELUTanhOp,
    OUT_INIT: Initializer = Zero,
] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    InitWith[Linear[U, BINS], OUT_INIT],
]

# Cont head MLP (1 hidden): feat[FEAT] → [Linear,RMSNorm,act] → logit[1]
comptime DreamerContMLP[FEAT: Int, U: Int, A: ElementOp = GELUTanhOp] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, 1],
]

# Value/slowvalue head (1 hidden): feat[FEAT] → twohot logits[BINS].
# Same shape as the reward MLP (symexp_twohot output); same structural
# OUT_INIT (paper zero-inits the critic output; reference `value.outscale: 0.0`).
comptime DreamerValue[
    FEAT: Int, U: Int, BINS: Int, A: ElementOp = GELUTanhOp,
    OUT_INIT: Initializer = Zero,
] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    InitWith[Linear[U, BINS], OUT_INIT],
]

# Policy head (1 hidden): feat[FEAT] → [mean_raw[ACT], std_raw[ACT]] = 2·ACT.
# `bounded_normal` (dists.mojo) maps mean_raw→tanh, std_raw→sigmoid-scaled.
comptime DreamerPolicy[FEAT: Int, U: Int, ACT: Int, A: ElementOp = GELUTanhOp] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, 2 * ACT],
]

# Discrete policy head (1 hidden): feat[FEAT] → logits[ACT] (ACT = #actions).
# `dists_discrete.mojo`'s unimix categorical (`OneHotDist`) maps logits→probs.
# Same MLP shape as the continuous head but a single ACT-wide logit output
# (vs 2·ACT mean/std). Used when the agent's `DISCRETE` flag is set.
comptime DreamerPolicyDiscrete[FEAT: Int, U: Int, ACT: Int, A: ElementOp = GELUTanhOp] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    Linear[U, ACT],
]

# Unified policy head selected by the `DISCRETE` flag. The output WIDTH is an
# Int comptime ternary (ACT logits if discrete, else 2·ACT mean/std), which
# resolves to a single concrete `Sequential` type — unlike a type-level
# ternary, which Mojo joins (breaking Movable). Use this when threading a
# compile-time `DISCRETE` flag through the trainer / AC block / agent.
comptime DreamerPolicyHead[
    FEAT: Int, U: Int, ACT: Int, DISCRETE: Bool, A: ElementOp = GELUTanhOp
] = Sequential[
    Linear[FEAT, U], RMSNorm[U], Elementwise[U, A],
    # Reference `policy.outscale = 0.01` declared STRUCTURALLY: near-zero
    # output logits → near-uniform initial policy that stays uniform until
    # real advantages arrive. Full-Kaiming logits are O(1) → a semi-collapsed
    # policy from step 0 that self-reinforces via its own replay (observed on
    # Pong: entropy 1.79→0.13 nats by 20k steps, all mass on one action
    # family). Fixed (not a knob) — it is reference parity, not a tuning axis.
    InitWith[Linear[U, ACT if DISCRETE else 2 * ACT], ScaledKaiming[1, 100]],
]
