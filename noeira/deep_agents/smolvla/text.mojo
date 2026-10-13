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
from noeira.nn.core.torch_names import TorchNameMap
from noeira.nn.quant.qlinear import Q4_INT4_GEMM
from noeira.io.safetensors import SafeTensors
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
    Q4: Bool = False,
](ParamWalkable):
    """16 layers plus the final norm, walked under the checkpoint's index.

    `KV_IN == W`: every VLM layer projects K/V from its own stream. (The expert
    is the one whose odd layers project the VLM's cached K/V instead.)
    `LORA_R > 0` puts a LoRA adapter beside each of every layer's seven linears
    (`layer_weights.mojo`). `Q4` makes the 112 linears int4 `QLinear`s, which
    `load_q4` fills; the walks then cover the norms (and adapters) only.
    """

    comptime Layer = DecoderLayerWeights[
        Self.W, Self.FF, Self.W, Self.KVW, Self.W, Self.LORA_R, Self.Q4
    ]
    comptime SCRATCH = Self.FF * Self.W
    """The largest linear's [OUT, IN]: what the shared dequantize scratch
    holds (gate / up / down, 2560 x 960)."""

    var layers: List[Self.Layer]
    var norm: SmolNorm[Self.W]
    var _q4_scratch: Bool
    """The int4 linears share one dequantize scratch (`ensure_q4_scratch`)."""

    def __init__(out self):
        self.layers = List[Self.Layer]()
        self.norm = SmolNorm[Self.W]()
        self._q4_scratch = False

    def __init__(out self, *, deinit move: Self):
        self.layers = move.layers^
        self.norm = move.norm^
        self._q4_scratch = move._q4_scratch

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

    # ── int4 ───────────────────────────────────────────────────────────
    def load_q4[
        target: StaticString
    ](
        mut self,
        ref st: SafeTensors,
        ref names: TorchNameMap,
        ctx: Optional[DeviceContext] = None,
        blocks_path: String = "",
    ) raises -> Int:
        """Quantize the 112 linears of the published weights `st` into the
        int4 layers (`DecoderLayerWeights.load_q4`), through the text name map;
        or, with `blocks_path`, read their Q4_0 blocks from that file
        (`load_q4_blocks`: GPTQ's output, `gptq_calib.mojo`). The norms load
        through the walk, as before. Returns the map entries claimed (each
        weight and its zero bias). On a GPU whose forward dequantizes (Metal,
        or `NN_Q4_PATH=deq`), the layers then share one dequantize scratch."""
        comptime assert Self.Q4, "load_q4: this tower is not int4 (Q4 = False)"
        var n = 0
        if blocks_path:
            var bf = SafeTensors(blocks_path)
            for i in range(Self.LAYERS):
                n += self.layers[i].load_q4_blocks[target](
                    st, names, String("layers.") + String(i), bf, ctx
                )
        else:
            for i in range(Self.LAYERS):
                n += self.layers[i].load_q4[target](
                    st, names, String("layers.") + String(i), ctx
                )
        comptime if target == "gpu" and not Q4_INT4_GEMM:
            self.ensure_q4_scratch(ctx.value())
        return n

    def ensure_q4_scratch(mut self, ctx: DeviceContext) raises:
        """One dequantize scratch for all 112 int4 linears (they run one at a
        time), allocated once. Every dequantizing call needs it: the forward
        off NVIDIA, the vjp everywhere."""
        comptime if Self.Q4:
            if self._q4_scratch:
                return
            var buf = ctx.enqueue_create_buffer[DT](Self.SCRATCH)
            for i in range(Self.LAYERS):
                self.layers[i].share_q4_scratch(buf, Self.SCRATCH)
            self._q4_scratch = True

    def keep_only_repacked(mut self) raises:
        """NVIDIA inference: every int4 linear keeps only MAX's repacked
        blocks (no host copy, no device copy of the logical blocks)."""
        for i in range(Self.LAYERS):
            self.layers[i].keep_only_repacked()
