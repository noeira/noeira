# +--------------------------------------------------------------------------+ #
# | SmolVLA — the action expert's weights (the fused loop's second stream)
# +--------------------------------------------------------------------------+ #
"""The 145 tensors of `model.vlm_with_expert.lm_expert`, in a container the
fused VLM+expert loop can drive.

## Why this is a weight container and not a `Module`

The expert is NOT a tower you can run after the VLM. `smolvlm_with_expert.py`
runs ONE loop over 16 layer indices, and at each index it touches both streams
and one shared attention:

    for i in range(16):
        h_p = vlm[i].input_layernorm(x_p)        # prefix stream, 960 wide
        h_s = expert[i].input_layernorm(x_s)     # suffix stream, 720 wide
        ... one attention, described below ...
        x_p = x_p + vlm[i].o_proj(a_p);   x_p = x_p + vlm[i].mlp(ln(x_p))
        x_s = x_s + expert[i].o_proj(a_s); x_s = x_s + expert[i].mlp(ln(x_s))

`Module` has one output, and this step has two streams; `Repeat`/`Sequential`
chain a single activation, and this chains a pair plus a per-layer KV cache. So
the loop is written by hand (next step) and the weights live here, walked by
index. What this file owns is the PARAMETERS and their names; nothing else.

## The three modes, and which layers do what

| mode           | streams            | per layer |
|----------------|--------------------|-----------|
| training       | `[prefix, suffix]` | joint self-attention over `cat(prefix, suffix)` |
| prefix prefill | `[prefix, None]`   | self-attention, **stores post-RoPE K/V per layer** |
| denoise step   | `[None, suffix]`   | even: append suffix K/V, attend `[prefix; suffix]`; odd: cross-attend the cached prefix |

After each denoise step the reference calls `past_key_values.crop(prefix_len)`,
dropping the suffix K/V the even layers appended so the next Euler step starts
from the same prefix. Ten steps, one prefill.

## Two layer kinds, and the shape that gives them away

`self_attn_every_n_layers = 2`, and the reference skips the reshape on
`layer_idx % 2 == 0`, so:

    even layers (0,2,…,14)  k,v : Linear[720 -> 320]   own stream  -> SELF
    odd  layers (1,3,…,15)  k,v : Linear[320 -> 320]   VLM's K/V   -> CROSS

Both kinds share q `Linear[720 -> 960]` and o `Linear[960 -> 720]`: the expert
always computes queries in the VLM's head geometry (15 heads x 64) and projects
the attention result back down to its own 720. That is what lets one attention
serve both streams — they meet in head space, not in width.

⚠ **Names mirror the checkpoint's own indexing** (`layers.7.self_attn.k.weight`),
not the walk's positional path. With two struct kinds held in two lists, a
positional name would number the eight self layers 0..7 and the eight cross
layers 0..7 again, and the map would have to undo that. Emitting the true index
keeps the map readable and the mistake visible.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.param import ParamVisitor, ParamWalkable
from noeira.nn.core.walkers import join_name
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.rms_norm import RMSNorm
from .layer_weights import DecoderLayerWeights, SmolNorm, DecoderMLP


comptime EXPERT_W: Int = 720          # expert_width_multiplier 0.75 * 960
comptime EXPERT_FF: Int = 2048
comptime EXPERT_LAYERS: Int = 16
comptime EXPERT_SELF_EVERY: Int = 2   # self_attn_every_n_layers
comptime VLM_W: Int = 960             # q/o meet the VLM here
comptime VLM_KV_W: Int = 320          # 5 kv heads x 64


struct SmolVLAExpert[
    LAYERS: Int = EXPERT_LAYERS,
    W: Int = EXPERT_W,
    FF: Int = EXPERT_FF,
    QW: Int = VLM_W,
    KVW: Int = VLM_KV_W,
    SELF_EVERY: Int = EXPERT_SELF_EVERY,
    LORA_R: Int = 0,
](ParamWalkable):
    """All 16 layers plus the final norm, walked under the CHECKPOINT's index.
    `LORA_R > 0` puts a LoRA adapter beside each of every layer's seven
    linears (`layer_weights.mojo`)."""

    comptime N_SELF: Int = (Self.LAYERS + Self.SELF_EVERY - 1) // Self.SELF_EVERY
    comptime N_CROSS: Int = Self.LAYERS - Self.N_SELF
    comptime SelfLayer = DecoderLayerWeights[
        Self.W, Self.FF, Self.QW, Self.KVW, Self.W, Self.LORA_R
    ]
    comptime CrossLayer = DecoderLayerWeights[
        Self.W, Self.FF, Self.QW, Self.KVW, Self.KVW, Self.LORA_R
    ]

    var self_layers: List[Self.SelfLayer]
    var cross_layers: List[Self.CrossLayer]
    var norm: SmolNorm[Self.W]

    def __init__(out self):
        self.self_layers = List[Self.SelfLayer]()
        self.cross_layers = List[Self.CrossLayer]()
        self.norm = SmolNorm[Self.W]()

    def __init__(out self, *, deinit move: Self):
        self.self_layers = move.self_layers^
        self.cross_layers = move.cross_layers^
        self.norm = move.norm^

    @staticmethod
    def is_self_layer(i: Int) -> Bool:
        """The reference's own test: `layer_idx % self_attn_every_n_layers == 0`
        keeps ordinary self-attention; every other index is cross-attention."""
        return i % Self.SELF_EVERY == 0

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var e = Self()
        for _ in range(Self.N_SELF):
            e.self_layers.append(Self.SelfLayer.make[target, INIT](ctx))
        for _ in range(Self.N_CROSS):
            e.cross_layers.append(Self.CrossLayer.make[target, INIT](ctx))
        e.norm = SmolNorm[Self.W].make[target, INIT](ctx)
        return e^

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut vis: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        # Emit the TRUE layer index, not the position within either list.
        for i in range(Self.LAYERS):
            var name = join_name(prefix, String("layers." + String(i)))
            if Self.is_self_layer(i):
                self.self_layers[i // Self.SELF_EVERY].walk[target](
                    vis, ctx, name
                )
            else:
                self.cross_layers[i // Self.SELF_EVERY].walk[target](
                    vis, ctx, name
                )
        self.norm.for_each_param[target](
            vis, ctx, join_name(prefix, String("norm"))
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """Every expert layer plus the final norm.

        ⚠ Walks `LAYERS` entries by TRUE layer index, exactly as
        `for_each_param` does, rather than the two lists' own lengths. Walking
        `self_layers` and `cross_layers` directly would give the same set here
        and would drift the moment the alternation changes.
        """
        for i in range(Self.LAYERS):
            if Self.is_self_layer(i):
                self.self_layers[i // Self.SELF_EVERY].zero_grad[target](ctx)
            else:
                self.cross_layers[i // Self.SELF_EVERY].zero_grad[target](ctx)
        self.norm.zero_grad[target](ctx)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut vis: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        """The State walk, in the SAME order as `for_each_param`.

        ⚠ Required by `ParamWalkable`, which the checkpoint needs and which
        also unlocks `Adam`'s grouped arena and its global `clip_grads` — the
        expert was the one component of the trainable set that was not one,
        so all three were blocked on this walk.

        No leaf here holds State today; see `DecoderMLP.walk_state` for why
        the walk exists anyway.
        """
        for i in range(Self.LAYERS):
            var name = join_name(prefix, String("layers." + String(i)))
            if Self.is_self_layer(i):
                self.self_layers[i // Self.SELF_EVERY].walk_state[target](
                    vis, ctx, name
                )
            else:
                self.cross_layers[i // Self.SELF_EVERY].walk_state[target](
                    vis, ctx, name
                )
        self.norm.for_each_state[target](
            vis, ctx, join_name(prefix, String("norm"))
        )
