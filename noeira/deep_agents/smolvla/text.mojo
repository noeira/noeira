# +--------------------------------------------------------------------------+ #
# | SmolVLA — the SmolLM2 text tower's weights
# +--------------------------------------------------------------------------+ #
"""The language half of SmolVLM2-500M, as shipped inside `lerobot/smolvla_base`.

    hidden 960 · 16 layers · 15 query heads over 5 KV heads (head_dim 64)
    SwiGLU ff 2560 · RMSNorm eps 1e-5 · rope_theta 100000 · vocab 49280

⚠ **16 layers, not 32.** The backbone's own `config.json` says
`num_hidden_layers = 32`; SmolVLA ships sixteen (`num_vlm_layers: 16`) and
truncates the tower to its first half. Depth comes from the checkpoint.

⚠ **The prefix pass is NOT causal.** `make_att_2d_masks` is prefix-LM and
SmolVLA sets `ar = 0` across BOTH the image and language spans, so the whole
visual+text prefix is one bidirectional block. See `attn_mask.mojo`.

## Weights, not a composed tower

This was first written as `Sequential[Repeat[16, layer], norm]`, which type-
checked and ran — and could not be used. The fused VLM+expert loop needs each
layer's K/V **between** the projection and the attention, to write into the
prefix cache, and a composed tower hands back only its final activation. So the
tower is a `DecoderLayerWeights` container, exactly like the expert, and the
loop that drives it lives in `fused.mojo`.

A VLM text layer IS an expert layer at different widths — same struct, five
different numbers — which is why both towers share `layer_weights.mojo` rather
than restating the topology twice.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.param import ParamVisitor, ParamWalkable
from noeira.nn.core.walkers import join_name
from noeira.nn.primitives.rms_norm import RMSNorm
from .layer_weights import DecoderLayerWeights, SmolNorm


comptime SMOLLM_DIM: Int = 960
comptime SMOLLM_HEADS: Int = 15
comptime SMOLLM_KV_HEADS: Int = 5
comptime SMOLLM_HEAD_DIM: Int = SMOLLM_DIM // SMOLLM_HEADS  # 64
comptime SMOLLM_REP: Int = SMOLLM_HEADS // SMOLLM_KV_HEADS  # 3
comptime SMOLLM_FF: Int = 2560
comptime SMOLLM_LAYERS: Int = 16
comptime SMOLLM_VOCAB: Int = 49280
comptime SMOLLM_THETA: Float64 = 100000.0
comptime SMOLLM_EPS: Float64 = 1e-5
comptime SMOLLM_KV_W: Int = SMOLLM_KV_HEADS * SMOLLM_HEAD_DIM  # 320


struct SmolVLMTextLayers[
    LAYERS: Int = SMOLLM_LAYERS,
    W: Int = SMOLLM_DIM,
    FF: Int = SMOLLM_FF,
    KVW: Int = SMOLLM_KV_W,
    LORA_R: Int = 0,
](ParamWalkable):
    """16 layers plus the final norm, walked under the checkpoint's index.

    `KV_IN == W`: every VLM layer projects K/V from its own stream. (The expert
    is the one whose odd layers project the VLM's cached K/V instead.)
    `LORA_R > 0` puts a LoRA adapter beside each of every layer's seven linears
    (`layer_weights.mojo`).
    """

    comptime Layer = DecoderLayerWeights[
        Self.W, Self.FF, Self.W, Self.KVW, Self.W, Self.LORA_R
    ]

    var layers: List[Self.Layer]
    var norm: SmolNorm[Self.W]

    def __init__(out self):
        self.layers = List[Self.Layer]()
        self.norm = SmolNorm[Self.W]()

    def __init__(out self, *, deinit move: Self):
        self.layers = move.layers^
        self.norm = move.norm^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var t = Self()
        for _ in range(Self.LAYERS):
            t.layers.append(Self.Layer.make[target, INIT](ctx))
        t.norm = SmolNorm[Self.W].make[target, INIT](ctx)
        return t^

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut vis: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        for i in range(Self.LAYERS):
            self.layers[i].walk[target](
                vis, ctx, join_name(prefix, String("layers." + String(i)))
            )
        self.norm.for_each_param[target](
            vis, ctx, join_name(prefix, String("norm"))
        )

    def for_each_base_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut vis: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        """`for_each_param` without the LoRA adapters: what the checkpoint's
        name map describes (`SmolVLAPolicy.load`) and what freezing the
        backbone freezes. The same walk as `for_each_param` at `LORA_R = 0`."""
        for i in range(Self.LAYERS):
            self.layers[i].walk[target, V, False](
                vis, ctx, join_name(prefix, String("layers." + String(i)))
            )
        self.norm.for_each_param[target](
            vis, ctx, join_name(prefix, String("norm"))
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """⚠ The SAME layers and norm as `for_each_param`. A frozen param has
        nothing to zero, so on a frozen tower this zeroes the adapters only."""
        for i in range(Self.LAYERS):
            self.layers[i].zero_grad[target](ctx)
        self.norm.zero_grad[target](ctx)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut vis: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        """The State walk, in the SAME order as `for_each_param`. No leaf here
        holds State; `ParamWalkable` (the optimizer arena, the checkpoints)
        asks for the walk, which LoRA on the tower needs."""
        for i in range(Self.LAYERS):
            self.layers[i].walk_state[target](
                vis, ctx, join_name(prefix, String("layers." + String(i)))
            )
        self.norm.for_each_state[target](
            vis, ctx, join_name(prefix, String("norm"))
        )

    def set_lora_alpha(mut self, alpha: Float64):
        """PEFT's `lora_alpha` for every adapter; nothing at `LORA_R = 0`."""
        for i in range(Self.LAYERS):
            self.layers[i].set_lora_alpha(alpha)
