# +--------------------------------------------------------------------------+ #
# | SmolVLA — the V1 inference policy: cameras + instruction -> action chunk
# +--------------------------------------------------------------------------+ #
"""Ties the pieces together into `sample_actions`.

    embed_prefix  ->  prefill (fills the KV cache)   ONCE
    embed_suffix  ->  denoise -> action_out          x10 Euler steps

⚠ **`lm_head` is not on this path.** It exists in the checkpoint and is mapped,
but SmolVLA's action head never uses it — the language modelling head is dead
weight for control. It is loaded only when something wants it.

⚠ **The prefix order is [images…, language…, state]** and the mask built from
`smolvla_ar` assumes exactly that. Assembling them in another order gives a
prefix of the same length whose block structure no longer matches the mask.
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from std.math import sqrt
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB
from noeira.nn.core.initializer import Initializer, Deterministic
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.silu import SiLU
from noeira.nn.primitives.pixel_shuffle import PixelShuffle
from noeira.nn.combinators.tokenwise import Tokenwise

from .vision import (
    SigLIPVisionTower, SIGLIP_GRID, SIGLIP_DIM, SIGLIP_IMG, SIGLIP_TOKENS,
    SIGLIP_PATCH, SIGLIP_HEADS, SIGLIP_FF, SIGLIP_LAYERS,
)
from .text import (
    SMOLLM_DIM, SMOLLM_KV_W, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, SMOLLM_LAYERS,
    SMOLLM_FF, SMOLLM_HEADS, SMOLLM_THETA, SmolVLMTextLayers,
)
from .heads import SMOLVLA_CONNECTOR_IN, SMOLVLA_EXPERT_W, SMOLVLA_ACTION_DIM
from .expert import SmolVLAExpert, EXPERT_FF
from .kv_cache import SmolVLAKVCache
from .fused import SmolVLADenoise, SmolVLAPrefill
from .flow import EulerSchedule, token_concat
from .embed import sinusoidal_time_embedding, embed_language_tokens
from .heads import SMOLVLA_STATE_DIM, SmolVLATokenEmbed
from .attn_mask import att_2d_mask, att_2d_mask_square, smolvla_ar
from .normalize import SmolVLAStats, normalize_state, unnormalize_action
from .names import (
    vision_name_map, text_name_map, expert_name_map, misc_name_map,
)
from noeira.io.safetensors import SafeTensors
from noeira.nn.core.torch_names import LoadTorchNamed
from noeira.nn.peft.freeze import freeze_params, FreezeParams


def _claimed_every_entry(
    what: String, filled: Int, entries: Int, skipped_by_design: Int = 0
) raises:
    """Every entry of the map was claimed by exactly one parameter.

    ⚠ This is the direction `report` cannot see. `report` raises when a
    PARAMETER has no map entry (`unmapped`) — the topology moved. The mirror
    failure is a map ENTRY that no parameter ever claimed, because the walk
    never emitted that name: the weight stays at its initialiser, the load
    reports success, and the policy runs and emits finite, plausible actions
    computed partly from noise. `Tokenwise` emitting `connector.0.weight`
    where the map says `connector.weight` is exactly this shape, and it is how
    that bug was found.

    ⚠ The expectation comes from the MAP, not from a number written here. A
    hardcoded 197 has to be edited whenever the map changes, and the edit that
    makes the gate pass again is the edit that stops it gating."""
    var want = entries - skipped_by_design
    if filled != want:
        raise Error(
            what + ": filled " + String(filled) + " of the map's "
            + String(entries) + " entries (" + String(skipped_by_design)
            + " skipped by design, so " + String(want) + " expected) — the"
            " unclaimed ones kept their initialisation, and nothing"
            " downstream can tell"
        )


def copy_into[
    target: StaticString
](
    mut dst: Tensor, dst_off: Int, mut src: Tensor, n: Int,
    ctx: Optional[DeviceContext] = None, src_off: Int = 0,
) raises:
    """Place `n` elements of `src[src_off:]` at `dst[dst_off:]` — prefix
    assembly's only primitive. Kept explicit rather than folded into a
    `Concat` because the prefix is built from a variable number of cameras.
    `src_off` is what lets a `[B, W]` state projection land row by row."""
    comptime if target == "cpu":
        for i in range(n):
            dst.data[dst_off + i] = src.data[src_off + i]
    else:
        var c = ctx.value()
        c.enqueue_copy(
            dst.dev.value().create_sub_buffer[DT](dst_off, n),
            src.dev.value().create_sub_buffer[DT](src_off, n),
        )


def scaled_copy_into[
    target: StaticString, N: Int, M: Int
](
    mut dst: Tensor, dst_off: Int, mut src: Tensor, scale: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`dst[dst_off:] <- src * scale` — the sqrt(960), fused into a copy that
    had to happen anyway.

    ⚠ **This replaces a GPU->CPU->GPU round trip per camera.** The scale used to
    be applied by `download`, a host multiply, then `upload` — and `upload`
    REALLOCATES the device buffer and calls `ctx.synchronize()` TWICE, so two
    cameras cost four synchronisations and two device allocations to multiply
    61,440 numbers by a constant. The copy into the prefix was already being
    made; the multiply now rides on it and costs nothing extra.
    """
    # ⚠ `M` is the DESTINATION's full length, not `N`. The kernel writes at
    # `dst_off + i`, so a view declared `row_major(N)` would be indexed past
    # its own extent — and a LayoutTensor index is int32 regardless of the
    # arithmetic that produced it.
    comptime assert N <= M, "scaled_copy_into: source longer than destination"
    if dst_off + N > M:
        raise Error(
            "scaled_copy_into: writing " + String(N) + " at " + String(dst_off)
            + " overruns a destination of " + String(M)
        )
    comptime if target == "cpu":
        for i in range(N):
            dst.data[dst_off + i] = src.data[i] * scale
    else:
        var c = ctx.value()
        comptime nb = (N + TPB - 1) // TPB
        c.enqueue_function[_scaled_copy_kernel[N, M]](
            dst.lt["gpu", Layout.row_major(M)](),
            src.lt["gpu", Layout.row_major(N)](),
            scale,
            # ⚠ `Int32`, not `Int`. `Int`/`UInt`/`Bool` are NOT `DevicePassable`
            # — a kernel taking one fails to instantiate, and the failure shows
            # up only when the kernel is actually built, not when the CPU path
            # of the same function type-checks.
            Int32(dst_off),
            grid_dim=nb,
            block_dim=TPB,
        )


def _scaled_copy_kernel[
    N: Int, M: Int
](
    dst: LayoutTensor[DT, Layout.row_major(M), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    scale: Scalar[DT],
    dst_off: Int32,
):
    var i = Int(global_idx.x)
    if i >= N:
        return
    dst.ptr[unsafe_offset = Int(dst_off) + i] = (
        rebind[Scalar[DT]](src.ptr[unsafe_offset=i]) * scale
    )


struct SmolVLAActionSampler[
    CHUNK: Int,
    ADIM: Int = SMOLVLA_ACTION_DIM,
    EW: Int = SMOLVLA_EXPERT_W,
    STEPS: Int = 10,
    B: Int = 1,
    LAYERS: Int = SMOLLM_LAYERS,
    # ⚠ Carried only so the sampler accepts a RECORDING denoise driver. The
    # ten Euler steps are the same computation either way — recording changes
    # where a layer writes, never what it computes — so a policy built for
    # training can still be asked for an action.
    REC: Bool = False,
    # The expert's LoRA rank (`layer_weights.mojo`), carried for the same
    # reason: the expert and its driver are typed by it.
    E_R: Int = 0,
](Movable):
    """The ten-step Euler loop, with `embed_suffix` and `action_out` inside it.

    Each step is: project the current noisy chunk, fuse it with the timestep,
    push it through the expert against the cached prefix, project back to action
    space, and take one negative Euler step.
    """

    comptime XN: Int = Self.B * Self.CHUNK * Self.ADIM
    comptime AN: Int = Self.B * Self.CHUNK * Self.EW
    comptime CN: Int = Self.B * Self.CHUNK * 2 * Self.EW
    comptime Sched = EulerSchedule[Self.STEPS]
    comptime Act = SiLU[Self.CHUNK * Self.EW]

    comptime XT = 0
    comptime AEMB = 1
    comptime TEMB = 2
    comptime CAT = 3
    comptime MID = 4
    comptime SUF = 5
    comptime OUT = 6
    comptime V = 7
    comptime N_SLOTS = 8

    var act: Self.Act
    var pool: TensorPack[Self.N_SLOTS]
    var temb: Tensor
    """All `STEPS` time embeddings, `[STEPS, EW]`, computed ONCE.

    ⚠ These are CONSTANTS. `time_at(step)` reads only the schedule, which is a
    comptime parameter, so the same ten vectors were being recomputed on the
    host and uploaded on EVERY query — and `upload` reallocates the device
    buffer and synchronises TWICE, so a ten-step denoise paid ten allocations
    and twenty full pipeline drains to transfer numbers that never change.
    Measured before this change: denoise was 76% of a 3.2 s query."""

    def __init__(out self):
        self.act = Self.Act()
        self.pool = TensorPack[Self.N_SLOTS]()
        self.temb = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.act = move.act^
        self.pool = move.pool^
        self.temb = move.temb^

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.act = Self.Act.make[target, Deterministic](ctx)
        # Build the whole schedule's embeddings here, upload once.
        s.temb = Tensor.alloc(Self.STEPS * Self.EW)
        for step in range(Self.STEPS):
            var te = sinusoidal_time_embedding[Self.EW](
                Self.Sched.time_at(step)
            )
            for i in range(Self.EW):
                s.temb.data[step * Self.EW + i] = te[i]
        comptime if target != "cpu":
            s.temb.upload(ctx.value())
        return s^

    def sample[
        target: StaticString, P: Int
    ](
        mut self,
        # ⚠ These name the canonical aliases directly rather than re-deriving
        # the widths. Mojo unifies comptime parameters by EXPRESSION, not by
        # value: `5 * (960 // 15)` and `320` are the same number and different
        # types, so every container in the loop must be spelled from the same
        # definitions.
        mut expert: SmolVLAExpert[
            Self.LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, 2, Self.E_R,
        ],
        mut cache: SmolVLAKVCache[
            Self.LAYERS, P, Self.CHUNK, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
            Self.B,
        ],
        mut denoise: SmolVLADenoise[
            P, Self.CHUNK, Self.B, Self.LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF,
            SMOLLM_DIM, SMOLLM_HEADS, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
            SMOLLM_THETA, 2, SMOLLM_KV_W, Self.REC, Self.E_R,
        ],
        mut action_in: Linear[Self.ADIM, Self.EW],
        mut time_mlp_in: Linear[2 * Self.EW, Self.EW],
        mut time_mlp_out: Linear[Self.EW, Self.EW],
        mut action_out: Linear[Self.EW, Self.ADIM],
        mut noise: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`noise` is x_1, `[B, CHUNK*ADIM]`; `out` receives x_0."""
        comptime TOK = Self.B * Self.CHUNK
        comptime if target == "cpu":
            self.pool[Self.XT].ensure(Self.XN)
            for i in range(Self.XN):
                self.pool[Self.XT].data[i] = noise.data[i]
        else:
            var c = ctx.value()
            self.pool[Self.XT].ensure_gpu(c, Self.XN)
            c.enqueue_copy(
                self.pool[Self.XT].dev.value().create_sub_buffer[DT](
                    0, Self.XN
                ),
                noise.dev.value().create_sub_buffer[DT](0, Self.XN),
            )

        for step in range(Self.STEPS):
            # ── embed_suffix ─────────────────────────────────────────────
            action_in.forward[target, TOK](
                TensorRefs[1](self.pool[Self.XT]), self.pool[Self.AEMB], ctx
            )
            # The time embedding is one vector, shared by every token —
            # `time_emb[:, None, :].expand_as(action_emb)`. Read from the
            # precomputed table: on GPU a device-to-device copy, so no host
            # round trip and no synchronisation inside the loop.
            comptime if target == "cpu":
                self.pool[Self.TEMB].ensure(Self.EW)
                for i in range(Self.EW):
                    self.pool[Self.TEMB].data[i] = self.temb.data[
                        step * Self.EW + i
                    ]
            else:
                var dc = ctx.value()
                self.pool[Self.TEMB].ensure_gpu(dc, Self.EW)
                dc.enqueue_copy(
                    self.pool[Self.TEMB].dev.value().create_sub_buffer[DT](
                        0, Self.EW
                    ),
                    self.temb.dev.value().create_sub_buffer[DT](
                        step * Self.EW, Self.EW
                    ),
                )
            # ⚠ per token: [action_t ‖ time], NOT [all actions ‖ all times]
            token_concat[target, Self.B, Self.CHUNK, Self.EW, Self.EW](
                self.pool[Self.AEMB], self.pool[Self.TEMB],
                self.pool[Self.CAT], ctx,
            )
            time_mlp_in.forward[target, TOK](
                TensorRefs[1](self.pool[Self.CAT]), self.pool[Self.MID], ctx
            )
            # in -> SiLU -> out: a plain two-layer MLP, NOT the decoder's SwiGLU
            self.act.forward[target, Self.B](
                TensorRefs[1](self.pool[Self.MID]), self.pool[Self.SUF], ctx
            )
            time_mlp_out.forward[target, TOK](
                TensorRefs[1](self.pool[Self.SUF]), self.pool[Self.OUT], ctx
            )

            # ── the expert, against the cached prefix ────────────────────
            denoise.step[target](
                expert, cache, self.pool[Self.OUT], self.pool[Self.SUF], ctx
            )
            action_out.forward[target, TOK](
                TensorRefs[1](self.pool[Self.SUF]), self.pool[Self.V], ctx
            )

            # ── one negative Euler step ──────────────────────────────────
            Self.Sched.advance[target, Self.XN](
                self.pool[Self.XT], self.pool[Self.V], ctx
            )

        comptime if target == "cpu":
            out.ensure(Self.XN)
            for i in range(Self.XN):
                out.data[i] = self.pool[Self.XT].data[i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, Self.XN)
            c.enqueue_copy(
                out.dev.value().create_sub_buffer[DT](0, Self.XN),
                self.pool[Self.XT].dev.value().create_sub_buffer[DT](
                    0, Self.XN
                ),
            )


struct SmolVLAPrefixEmbed[
    N_CAM: Int,
    N_LANG: Int,
    B: Int = 1,
    W: Int = SMOLLM_DIM,
    IMG_TOK: Int = 64,
    VIS_LAYERS: Int = SIGLIP_LAYERS,
](Movable):
    """`[images…, language…, state]` -> `[B, P * 960]`, the VLM stream.

    Per camera: SigLIP tower -> PixelShuffle -> connector -> **x sqrt(960)**.
    Language: an embedding row gather, also **x sqrt(960)**. State: `state_proj`,
    **not** scaled.

    ⚠ **The order is load-bearing.** `smolvla_ar(N_CAM*IMG_TOK, N_LANG, 1, chunk)`
    describes exactly this layout, and the prefill mask is built from it. Writing
    the segments in another order gives a prefix of the same length whose block
    structure no longer matches its mask — finite, right-shaped, wrong.

    ⚠ **The sqrt(960) applies AFTER the connector**, because the reference takes
    `img_emb_dim` from `embed_image`'s output (960), not from the vision tower's
    768. Scaling by sqrt(768) would be a plausible misreading and ~14% off.

    ⚠ The `add_image_special_tokens = False` branch is assumed. Enabling it would
    put a start and an end token around every camera, changing `P` and every mask
    derived from it.
    """

    comptime IMG_N: Int = Self.IMG_TOK * Self.W
    comptime IMG_SEG: Int = Self.N_CAM * Self.IMG_N
    """The image segment: every camera's 64 tokens, the part of the prefix a
    frame determines on its own and the unit `vision_cache.mojo` stores."""
    comptime P: Int = Self.N_CAM * Self.IMG_TOK + Self.N_LANG + 1
    comptime OUT_N: Int = Self.B * Self.P * Self.W
    comptime VIS_IN: Int = 3 * SIGLIP_IMG * SIGLIP_IMG
    comptime VIS_OUT: Int = SIGLIP_TOKENS * SIGLIP_DIM
    comptime Shuffle = PixelShuffle[SIGLIP_GRID, SIGLIP_DIM, 4]
    comptime Vision = SigLIPVisionTower[
        SIGLIP_IMG, SIGLIP_PATCH, SIGLIP_DIM, SIGLIP_HEADS, SIGLIP_FF,
        Self.VIS_LAYERS, SIGLIP_GRID, SIGLIP_TOKENS,
    ]
    """⚠ Only the DEPTH is reducible. `IMG`/`GRID`/`TOKENS` are load-bearing
    downstream — `PixelShuffle[32, 768, 4]` and `IMG_TOK = 64` are derived from
    the 32x32 patch grid — so a smaller image is not a cheaper test, it is a
    different architecture."""

    var shuffle: Self.Shuffle
    var one: Tensor
    """One camera's pixels, allocated ONCE.

    ⚠ This used to be `Tensor.alloc(VIS_IN)` inside the camera loop — 3.15 MB
    per camera per call, so 6.3 MB of allocation churn per inference at two
    cameras, plus an `upload` that reallocates the device buffer and
    synchronises twice."""
    var vis: Tensor
    var shuf: Tensor
    var conn: Tensor
    var lang: Tensor
    var st: Tensor

    def __init__(out self):
        comptime assert Self.N_CAM >= 1, "SmolVLAPrefixEmbed: need a camera"
        self.shuffle = Self.Shuffle()
        self.one = Tensor()
        self.vis = Tensor()
        self.shuf = Tensor()
        self.conn = Tensor()
        self.lang = Tensor()
        self.st = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.shuffle = move.shuffle^
        self.one = move.one^
        self.vis = move.vis^
        self.shuf = move.shuf^
        self.conn = move.conn^
        self.lang = move.lang^
        self.st = move.st^

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var p = Self()
        p.shuffle = Self.Shuffle.make[target, Deterministic](ctx)
        comptime if target == "cpu":
            p.one = Tensor.alloc(Self.VIS_IN)
        else:
            p.one = Tensor.alloc_gpu(ctx.value(), Self.VIS_IN)
        return p^

    def run[
        target: StaticString, VOCAB: Int, CONN_IN: Int, SDIM: Int
    ](
        mut self,
        mut vision: Self.Vision,
        mut connector: Tokenwise[Self.IMG_TOK, Linear[CONN_IN, Self.W]],
        mut embed_weight: Tensor,
        mut state_proj: Linear[SDIM, Self.W],
        mut images: Tensor,
        ref lang_ids: List[Int],
        mut state: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`images` is `[N_CAM, 3*512*512]` (batch 1 per camera, as SmolVLA
        runs them); `lang_ids` are PRE-TOKENISED ids.

        Two halves, callable separately: `run_images` writes the image
        segment, `run_tail` the language and state tokens after it. The
        fine-tune's vision cache (`vision_cache.mojo`) stores what the first
        half produces and feeds `run_tail` directly.
        """
        self.run_images[target, CONN_IN](vision, connector, images, out, ctx)
        self.run_tail[target, VOCAB, SDIM](
            embed_weight, state_proj, lang_ids, state, out, ctx
        )

    def run_images[
        target: StaticString, CONN_IN: Int
    ](
        mut self,
        mut vision: Self.Vision,
        mut connector: Tokenwise[Self.IMG_TOK, Linear[CONN_IN, Self.W]],
        mut images: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`out[0 : IMG_SEG]` <- every camera's scaled connector output.

        ⚠ **On GPU, `images` must ALREADY BE DEVICE-RESIDENT.** Each camera's
        slab is taken with a device-to-device sub-buffer copy, which is what
        makes it free; a host-only tensor would have to be uploaded per camera,
        which is the cost this path exists to avoid. `fill_camera_images`
        uploads, so the real producer already satisfies this. The check below
        makes the requirement a named error rather than an `Optional.value()`
        abort deep in the loop.
        """
        comptime if target != "cpu":
            if not images.dev:
                raise Error(
                    "SmolVLAPrefixEmbed.run: `images` has no device buffer."
                    " On GPU the per-camera slab is a device-to-device copy, so"
                    " the caller must upload first —"
                    " `fill_camera_images` does."
                )
        comptime if target == "cpu":
            out.ensure(Self.OUT_N)
        else:
            out.ensure_gpu(ctx.value(), Self.OUT_N)

        var scale = sqrt(Scalar[DT](Self.W))
        var off = 0

        # ── cameras ──────────────────────────────────────────────────────
        for cam in range(Self.N_CAM):
            # One camera's pixels out of the back-to-back block. On GPU this is
            # a DEVICE-TO-DEVICE sub-buffer copy: no allocation, no host round
            # trip, and none of `upload`'s two synchronisations.
            comptime if target == "cpu":
                for i in range(Self.VIS_IN):
                    self.one.data[i] = images.data[cam * Self.VIS_IN + i]
            else:
                var c = ctx.value()
                c.enqueue_copy(
                    self.one.dev.value().create_sub_buffer[DT](
                        0, Self.VIS_IN
                    ),
                    images.dev.value().create_sub_buffer[DT](
                        cam * Self.VIS_IN, Self.VIS_IN
                    ),
                )
            vision.forward[target, 1](TensorRefs[1](self.one), self.vis, ctx)
            self.shuffle.forward[target, 1](
                TensorRefs[1](self.vis), self.shuf, ctx
            )
            connector.forward[target, 1](
                TensorRefs[1](self.shuf), self.conn, ctx
            )
            # ⚠ sqrt of the CONNECTOR's width (960), not the tower's (768),
            # fused into the copy rather than applied by a round trip.
            scaled_copy_into[target, Self.IMG_N, Self.OUT_N](
                out, off, self.conn, scale, ctx
            )
            off += Self.IMG_N
        if off != Self.IMG_SEG:
            raise Error(
                "SmolVLAPrefixEmbed.run_images: wrote " + String(off)
                + " of " + String(Self.IMG_SEG)
            )

    def run_tail[
        target: StaticString, VOCAB: Int, SDIM: Int
    ](
        mut self,
        mut embed_weight: Tensor,
        mut state_proj: Linear[SDIM, Self.W],
        ref lang_ids: List[Int],
        mut state: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`out[b, IMG_SEG : P*W]` <- language tokens, then the state token,
        for every batch row `b`.

        Assumes the image segment is ALREADY in `out` — from `run_images` or
        from cache rows uploaded by `SmolVLAPolicy.build_prefix_from_segment`.

        `lang_ids` is either ONE instruction (`N_LANG` ids, shared by every
        row — the single-task recording) or `B * N_LANG` ids, one instruction
        per row. `state` is `[B, SDIM]`.
        """
        var per_row = len(lang_ids) == Self.B * Self.N_LANG
        if not per_row and len(lang_ids) != Self.N_LANG:
            raise Error(
                "SmolVLAPrefixEmbed: expected " + String(Self.N_LANG)
                + " language ids (shared) or " + String(Self.B * Self.N_LANG)
                + " (one instruction per row), got " + String(len(lang_ids))
            )
        comptime if target == "cpu":
            out.ensure(Self.OUT_N)
        else:
            out.ensure_gpu(ctx.value(), Self.OUT_N)
        comptime ROW = Self.P * Self.W
        comptime LANG_N = Self.N_LANG * Self.W

        # ── state: projected for the whole batch, and NOT scaled ─────────
        state_proj.forward[target, Self.B](
            TensorRefs[1](state), self.st, ctx
        )

        for b in range(Self.B):
            var off = b * ROW + Self.IMG_SEG
            # ── language: a row gather, scaled the same way ──────────────
            # Gathered once when shared, per row otherwise. The gather is
            # host-side, so its transfer stays; `upload_resident` keeps the
            # buffer rather than recreating it per row.
            if per_row or b == 0:
                var ids = List[Int]()
                var base = b * Self.N_LANG if per_row else 0
                for t in range(Self.N_LANG):
                    ids.append(lang_ids[base + t])
                embed_language_tokens[VOCAB, Self.W](
                    embed_weight, ids, self.lang, True
                )
                comptime if target != "cpu":
                    self.lang.upload_resident(ctx.value())
            copy_into[target](out, off, self.lang, LANG_N, ctx)
            off += LANG_N
            # ── state token: row b of the projection ─────────────────────
            copy_into[target](out, off, self.st, Self.W, ctx, b * Self.W)
            off += Self.W
            if off != (b + 1) * ROW:
                raise Error(
                    "SmolVLAPrefixEmbed: row " + String(b) + " wrote to "
                    + String(off) + ", expected " + String((b + 1) * ROW)
                    + " — the segment widths do not sum to P*W"
                )


struct SmolVLAPolicy[
    N_CAM: Int,
    N_LANG: Int,
    CHUNK: Int = 50,
    STEPS: Int = 10,
    B: Int = 1,
    LAYERS: Int = SMOLLM_LAYERS,
    VIS_LAYERS: Int = SIGLIP_LAYERS,
    # V2. False is inference: one scratch pool per driver, nothing kept. True
    # makes both drivers record their per-layer activations so `backward` can
    # walk them. It costs ~320 MB of tape at the published depths and changes
    # no arithmetic — `test_denoise_tape.mojo` demands the outputs be
    # bit-identical either way.
    RECORD: Bool = False,
    # LoRA ranks (`nn/peft/lora.mojo`, `layer_weights.mojo`): an adapter
    # beside every linear of the expert / of the 16 text layers. 0, the
    # default, builds none and is today's policy exactly. The base weights
    # load and freeze as before (`load`, `freeze_backbone` walk them without
    # the adapters); the adapters start at PEFT's initialisation, a no-op.
    EXPERT_R: Int = 0,
    TOWER_R: Int = 0,
    # The VLM's 16 text layers in int4 (`nn/quant`, Q4_0): `load` quantizes
    # their 112 linears from the published weights by round-to-nearest, and
    # the prefill runs them through MAX's int4 GEMM on NVIDIA. False, the
    # default, is today's fp32 tower exactly.
    VLM_Q4: Bool = False,
](Movable):
    """Every component of SmolVLA in one object: pixels and a pose in, joint
    angles out.

    The pieces have all existed and all been gated separately. What did not
    exist is the thing that owns them together, and that is where the errors
    a component gate cannot see live: a mask built for one `P` and a prefix
    built for another, a chunk unnormalised with the state's statistics, a
    cache prefilled once and reused across two different observations.

        raw frames  -> resize_with_pad -> SigLIP -> shuffle -> connector ─┐
        instruction -> pre-tokenised ids -> embedding ────────────────────┤-> prefix
        raw qpos    -> (x-mean)/std -> pad 32 -> state_proj ──────────────┘
                    -> prefill 16 layers -> KV cache
                    -> 10 Euler steps through the expert
                    -> action_out -> drop dims 6.. -> x*std+mean

    ⚠ **`P` is computed here and nowhere else.** `N_CAM*64 + N_LANG + 1` feeds
    the cache, the prefill mask, the denoise masks and the prefix buffer at
    once. Every one of them is a plain integer parameter, so a component built
    from a hand-written `P` that disagrees produces finite output with a mask
    describing a different sequence. Deriving it once is the only defence.

    ⚠ **`lm_head` is deliberately NOT owned.** SmolVLA never generates text --
    the VLM contributes a KV cache, not tokens -- and the head is
    `[49280, 960]`, 47.3 M parameters, 189 MB. The name map still claims it so
    the 500/500 coverage gate stays honest; this object simply does not
    instantiate it, and `load` reports the count it expects rather than letting
    the omission be silent.

    ⚠ **The cache belongs to ONE observation.** `select_action` refills it every
    call. Prefilling once and sampling repeatedly would run the expert against a
    stale scene: no error, no NaN, just a policy acting on what it saw before.

    ⚠ **`LAYERS` / `VIS_LAYERS` exist for TEST FIXTURES ONLY.** At the published
    depths one policy is 402,737,376 params x 4 bytes x 2 (`Tensor` keeps a host
    AND a device copy) = **3.2 GB**, which is more than a wiring gate should
    need. A shallow build exercises the same wiring for a fraction of it.
    `load` REFUSES anything but the real depths — see its assertions. Only the
    depth is reducible: image size, widths and the vocabulary are load-bearing
    downstream, so a smaller one of those is a different architecture rather
    than a cheaper test.
    """

    comptime IMG_TOK: Int = 64
    comptime P: Int = Self.N_CAM * Self.IMG_TOK + Self.N_LANG + 1
    comptime W: Int = SMOLLM_DIM
    comptime EW: Int = SMOLVLA_EXPERT_W
    comptime L: Int = Self.LAYERS
    comptime ADIM: Int = SMOLVLA_ACTION_DIM
    comptime SDIM: Int = SMOLVLA_STATE_DIM
    comptime VOCAB: Int = 49280

    comptime Tower = SmolVLMTextLayers[
        Self.L, Self.W, SMOLLM_FF, SMOLLM_KV_W, Self.TOWER_R, Self.VLM_Q4
    ]
    comptime Expert = SmolVLAExpert[
        Self.L, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM, SMOLLM_KV_W, 2,
        Self.EXPERT_R,
    ]
    comptime Cache = SmolVLAKVCache[
        Self.L, Self.P, Self.CHUNK, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, Self.B
    ]
    comptime Pre = SmolVLAPrefill[
        Self.P, Self.CHUNK, Self.B, Self.LAYERS, SMOLLM_DIM, SMOLLM_FF,
        SMOLLM_HEADS, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, SMOLLM_THETA,
        Self.RECORD, Self.TOWER_R, Self.VLM_Q4,
    ]
    comptime Den = SmolVLADenoise[
        Self.P, Self.CHUNK, Self.B, Self.LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF,
        SMOLLM_DIM, SMOLLM_HEADS, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
        SMOLLM_THETA, 2, SMOLLM_KV_W, Self.RECORD, Self.EXPERT_R,
    ]
    comptime Sam = SmolVLAActionSampler[
        Self.CHUNK, SMOLVLA_ACTION_DIM, SMOLVLA_EXPERT_W, Self.STEPS, Self.B,
        Self.LAYERS, Self.RECORD, Self.EXPERT_R,
    ]
    comptime Prefix = SmolVLAPrefixEmbed[
        Self.N_CAM, Self.N_LANG, Self.B, SMOLLM_DIM, Self.IMG_TOK,
        Self.VIS_LAYERS,
    ]
    comptime Conn = Tokenwise[
        Self.IMG_TOK, Linear[SMOLVLA_CONNECTOR_IN, SMOLLM_DIM]
    ]
    comptime Embed = SmolVLATokenEmbed[Self.VOCAB, SMOLLM_DIM]

    var vision: Self.Prefix.Vision
    var connector: Self.Conn
    var embed: Self.Embed
    var tower: Self.Tower
    var expert: Self.Expert
    var state_proj: Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM]
    var action_in: Linear[SMOLVLA_ACTION_DIM, SMOLVLA_EXPERT_W]
    var time_mlp_in: Linear[2 * SMOLVLA_EXPERT_W, SMOLVLA_EXPERT_W]
    var time_mlp_out: Linear[SMOLVLA_EXPERT_W, SMOLVLA_EXPERT_W]
    var action_out: Linear[SMOLVLA_EXPERT_W, SMOLVLA_ACTION_DIM]

    var cache: Self.Cache
    var prefill: Self.Pre
    var denoiser: Self.Den
    var sampler: Self.Sam
    var prefix: Self.Prefix
    var stats: SmolVLAStats

    var prefix_buf: Tensor
    var prefill_out: Tensor
    var state_buf: Tensor
    var chunk_buf: Tensor

    def __init__(out self):
        comptime assert Self.N_CAM >= 1, "SmolVLAPolicy: need a camera"
        comptime assert Self.N_LANG >= 1, "SmolVLAPolicy: need an instruction"
        self.vision = Self.Prefix.Vision()
        self.connector = Self.Conn()
        self.embed = Self.Embed()
        self.tower = Self.Tower()
        self.expert = Self.Expert()
        self.state_proj = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM]()
        self.action_in = Linear[SMOLVLA_ACTION_DIM, SMOLVLA_EXPERT_W]()
        self.time_mlp_in = Linear[2 * SMOLVLA_EXPERT_W, SMOLVLA_EXPERT_W]()
        self.time_mlp_out = Linear[SMOLVLA_EXPERT_W, SMOLVLA_EXPERT_W]()
        self.action_out = Linear[SMOLVLA_EXPERT_W, SMOLVLA_ACTION_DIM]()
        self.cache = Self.Cache()
        self.prefill = Self.Pre()
        self.denoiser = Self.Den()
        self.sampler = Self.Sam()
        self.prefix = Self.Prefix()
        self.stats = SmolVLAStats()
        self.prefix_buf = Tensor()
        self.prefill_out = Tensor()
        self.state_buf = Tensor()
        self.chunk_buf = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.vision = move.vision^
        self.connector = move.connector^
        self.embed = move.embed^
        self.tower = move.tower^
        self.expert = move.expert^
        self.state_proj = move.state_proj^
        self.action_in = move.action_in^
        self.time_mlp_in = move.time_mlp_in^
        self.time_mlp_out = move.time_mlp_out^
        self.action_out = move.action_out^
        self.cache = move.cache^
        self.prefill = move.prefill^
        self.denoiser = move.denoiser^
        self.sampler = move.sampler^
        self.prefix = move.prefix^
        self.stats = move.stats^
        self.prefix_buf = move.prefix_buf^
        self.prefill_out = move.prefill_out^
        self.state_buf = move.state_buf^
        self.chunk_buf = move.chunk_buf^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer = Deterministic
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Build every component, with the three masks derived from one `P`.

        `INIT` seeds the weights; `load` replaces them. A gate that never calls
        `load` still runs, which is what lets the wiring be checked without a
        907 MB download."""
        var p = Self()
        # ⚠ ONE `ar` per window, all from the same layout. `smolvla_ar`'s
        # arguments are the segment lengths this object also uses to size the
        # prefix buffer, so a disagreement is impossible rather than unlikely.
        var img = Self.N_CAM * Self.IMG_TOK
        var ar_pre = smolvla_ar(img, Self.N_LANG, 1, 0)
        var ar_full = smolvla_ar(img, Self.N_LANG, 1, Self.CHUNK)
        var mask_pre = att_2d_mask_square(ar_pre)
        var mask_self = att_2d_mask(
            ar_full, Self.P, Self.P + Self.CHUNK, 0, Self.P + Self.CHUNK
        )
        var mask_cross = att_2d_mask(
            ar_full, Self.P, Self.P + Self.CHUNK, 0, Self.P
        )

        p.vision = Self.Prefix.Vision.make[target, INIT](ctx)
        p.connector = Self.Conn.make[target, INIT](ctx)
        p.embed = Self.Embed.make[target, INIT](ctx)
        p.tower = Self.Tower.make[target, INIT](ctx)
        p.expert = Self.Expert.make[target, INIT](ctx)
        p.state_proj = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM].make[
            target, INIT
        ](ctx)
        p.action_in = Linear[SMOLVLA_ACTION_DIM, SMOLVLA_EXPERT_W].make[
            target, INIT
        ](ctx)
        p.time_mlp_in = Linear[
            2 * SMOLVLA_EXPERT_W, SMOLVLA_EXPERT_W
        ].make[target, INIT](ctx)
        p.time_mlp_out = Linear[SMOLVLA_EXPERT_W, SMOLVLA_EXPERT_W].make[
            target, INIT
        ](ctx)
        p.action_out = Linear[SMOLVLA_EXPERT_W, SMOLVLA_ACTION_DIM].make[
            target, INIT
        ](ctx)
        p.cache = Self.Cache.make[target](ctx)
        p.prefill = Self.Pre.make[target](mask_pre, ctx)
        p.denoiser = Self.Den.make[target](mask_self, mask_cross, ctx)
        p.sampler = Self.Sam.make[target](ctx)
        p.prefix = Self.Prefix.make[target](ctx)
        return p^

    def set_fused_vision_attention(mut self, on: Bool):
        """Route the SigLIP towers' attention through `CrossAttention`'s
        fused inference kernel (no scores, no cache, no packs): ~130 ms off a
        query on the Orin. INFERENCE ONLY — a vjp through the tower raises
        while it is on, and its output differs from the two-pass kernel at
        the last bit, so the fine-tune (and its vision cache) leaves it off.
        The deploy turns it on after `load`."""
        self.vision.set_attr["fused_attention"](
            Scalar[DT](1.0) if on else Scalar[DT](0.0)
        )

    def load_stats(mut self, stats_json: String) raises:
        """`<dataset>/meta/stats.json` — the fine-tune's own normalisation."""
        self.stats = SmolVLAStats.from_stats_json(stats_json)

    def load[
        target: StaticString
    ](
        mut self, weights: String, ctx: Optional[DeviceContext] = None
    ) raises:
        """Fill every owned component from `model.safetensors`.

        Four maps, four walks. The counts are ASSERTED rather than printed: a
        map that quietly matched 140 of 145 tensors leaves five layers at their
        initialiser, which is a policy that runs, produces finite actions, and
        is wrong in a way no downstream check can attribute.
        """
        # ⚠ `report`, not `report_exhaustive`/`report_exact`: those two check
        # coverage of the WHOLE FILE, which is right for a map that claims all
        # 500 tensors and wrong for one component of four. Here the file-side
        # question is answered once, by `test_checkpoint_coverage`; what each
        # walk must answer is its own — nothing unmapped, nothing missing, and
        # the count it was supposed to fill.
        # ⚠ A REDUCED-DEPTH POLICY CANNOT HOLD THIS CHECKPOINT, and must say
        # so here rather than fail downstream. The name maps enumerate all 16
        # VLM / 16 expert / 12 vision layers, so a shallower walk leaves most
        # entries unclaimed — `_claimed_every_entry` would catch it, but with a
        # message about counts rather than about depth. Worse, the tempting
        # "fix" is to parameterise the maps too, which would silently give a
        # policy holding the FIRST TWO of sixteen layers: it runs, it is
        # finite, and it emits plausible actions.
        comptime assert Self.LAYERS == SMOLLM_LAYERS, (
            "SmolVLAPolicy.load: this policy was built with a reduced LAYERS."
            " A shallow policy is a TEST FIXTURE — it cannot hold"
            " lerobot/smolvla_base."
        )
        comptime assert Self.VIS_LAYERS == SIGLIP_LAYERS, (
            "SmolVLAPolicy.load: this policy was built with a reduced"
            " VIS_LAYERS. A shallow vision tower is a TEST FIXTURE — it cannot"
            " hold lerobot/smolvla_base."
        )

        var vl = LoadTorchNamed[""](SafeTensors(weights), vision_name_map())
        var _ref1 = ParamVisitorRef.of[type_of(vl), target](vl)
        self.vision.for_each_param[target](_ref1, ctx)
        vl.report(String("vision"))
        _claimed_every_entry(
            String("vision"), len(vl.loaded) + len(vl.zeroed),
            len(vl.map.ours),
        )

        # ⚠ The BASE walks for the two towers: the name maps describe the
        # checkpoint, which has no LoRA adapters, and `report` raises on a
        # parameter the map does not name. Identical walks at rank 0.
        var tl = LoadTorchNamed[""](SafeTensors(weights), text_name_map())
        var _ref2 = ParamVisitorRef.of[type_of(tl), target](tl)
        self.tower.for_each_base_param[target](_ref2, ctx)
        tl.report(String("text"))
        # An int4 tower's walk is its norms; its 112 linears are quantized
        # from the same file, through the same map, and claim their entries
        # (each weight and its zero bias) here.
        var q4_claimed = 0
        comptime if Self.VLM_Q4:
            q4_claimed = self.tower.load_q4[target](tl.file, tl.map, ctx)
        _claimed_every_entry(
            String("text"), len(tl.loaded) + len(tl.zeroed) + q4_claimed,
            len(tl.map.ours),
        )

        var el = LoadTorchNamed[""](SafeTensors(weights), expert_name_map())
        var _ref3 = ParamVisitorRef.of[type_of(el), target](el)
        self.expert.for_each_base_param[target](_ref3, ctx)
        el.report(String("expert"))
        _claimed_every_entry(
            String("expert"), len(el.loaded) + len(el.zeroed),
            len(el.map.ours),
        )

        # ⚠ The misc map addresses seven separate small modules by dotted
        # prefix, so each is walked with the name the map uses. `lm_head` is
        # absent on purpose — see the struct header — which is why this reports
        # 12 of the map's 13 file-backed entries.
        var ml = LoadTorchNamed[""](SafeTensors(weights), misc_name_map())
        # ⚠ `.inner`, not the wrapper. `Tokenwise.for_each_param` appends its
        # position, emitting `connector.0.weight`, while `misc_name_map` names
        # `connector.weight` — the combinator is a compile-time detail of HOW
        # the connector is applied per token, not part of the checkpoint's
        # naming. Walking the wrapper leaves the connector at its initialiser
        # and reports it as `unmapped`.
        var _ref4 = ParamVisitorRef.of[type_of(ml), target](ml)
        self.connector.inner.for_each_param[target](_ref4, ctx, String("connector")
        )
        var _ref5 = ParamVisitorRef.of[type_of(ml), target](ml)
        self.embed.for_each_param[target](_ref5, ctx, String("embed"))
        var _ref6 = ParamVisitorRef.of[type_of(ml), target](ml)
        self.state_proj.for_each_param[target](_ref6, ctx, String("state_proj"))
        var _ref7 = ParamVisitorRef.of[type_of(ml), target](ml)
        self.action_in.for_each_param[target](_ref7, ctx, String("action_in"))
        var _ref8 = ParamVisitorRef.of[type_of(ml), target](ml)
        self.action_out.for_each_param[target](_ref8, ctx, String("action_out"))
        var _ref9 = ParamVisitorRef.of[type_of(ml), target](ml)
        self.time_mlp_in.for_each_param[target](_ref9, ctx, String("time_mlp_in")
        )
        var _ref10 = ParamVisitorRef.of[type_of(ml), target](ml)
        self.time_mlp_out.for_each_param[target](_ref10, ctx, String("time_mlp_out")
        )
        ml.report(String("heads"))
        # ⚠ TWO entries skipped by design — `lm_head.weight` (file-backed) and
        # `lm_head.bias` (TN_ZEROS). Naming the number is what keeps the
        # omission a DECISION; without it, `lm_head` and any FUTURE component
        # someone forgets to walk are indistinguishable.
        _claimed_every_entry(
            String("heads"), len(ml.loaded) + len(ml.zeroed),
            len(ml.map.ours), 2,
        )
        print(
            "  policy: "
            + String(len(vl.loaded) + len(tl.loaded) + len(el.loaded)
                     + len(ml.loaded))
            + " tensors loaded (lm_head's 1 skipped by design)"
        )

    def freeze_backbone[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext] = None) raises -> Int:
        """Freeze the vision tower, the connector, the token embedding and
        the 16 text layers: what `train_expert_only` never trains, minus
        `state_proj`, whose regime the caller picks. Returns the number of
        values frozen.

        Their gradient buffers (two copies each on a GPU: host and device)
        are released, and the text tower's backward, which `train_state_proj`
        runs, then forms grad_input only (`Linear`, `RMSNorm`). The vision
        tower, connector and embedding are upstream of nothing trained, so
        their backward never runs. Values, names and checkpoint maps are
        unchanged.

        ⚠ Before the optimizer adopts anything; see `noeira/nn/peft/freeze.mojo`.

        With `TOWER_R > 0` the text layers' LoRA adapters are NOT frozen: the
        tower is frozen by its base walk, which is everything it froze before.
        """
        var n = freeze_params[target](self.vision, ctx)
        n += freeze_params[target](self.connector, ctx)
        n += freeze_params[target](self.embed, ctx)
        var f = FreezeParams()
        self.tower.for_each_base_param[target](f, ctx)
        return n + f.values

    def freeze_expert_base[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext] = None) raises -> Int:
        """Freeze the expert's own weights (its 16 layers' linears and norms
        and the final norm) and leave its LoRA adapters trainable: LoRA on the
        expert. Returns the number of values frozen. Before any adopt."""
        comptime assert Self.EXPERT_R > 0, (
            "freeze_expert_base: without adapters this freezes the whole"
            " expert, which is freeze_all's job"
        )
        var f = FreezeParams()
        self.expert.for_each_base_param[target](f, ctx)
        return f.values

    def freeze_all[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext] = None) raises -> Int:
        """Freeze every parameter: the deploy, where nothing trains. Returns
        the number of values frozen."""
        var n = self.freeze_backbone[target](ctx)
        n += freeze_params[target](self.expert, ctx)
        n += freeze_params[target](self.state_proj, ctx)
        n += freeze_params[target](self.action_in, ctx)
        n += freeze_params[target](self.time_mlp_in, ctx)
        n += freeze_params[target](self.time_mlp_out, ctx)
        n += freeze_params[target](self.action_out, ctx)
        return n

    def build_prefix[
        target: StaticString
    ](
        mut self,
        mut images: Tensor,
        ref lang_ids: List[Int],
        ref raw_state: List[Float32],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One observation to a filled KV cache. Everything before the action.

        Inference calls this and then integrates; TRAINING calls it and then
        takes one denoising step at a sampled `t`. It is factored out so the
        two share a single description of what a prefix IS — the state
        normalisation, the segment order, the cache reset — rather than a
        training driver growing its own copy that drifts from this one. That
        drift is the defect shape this port has paid for most.

        ⚠ The cache is RESET every call. A cache carried across observations
        is a policy acting on the previous scene, silently — and in training
        it is a batch element conditioned on its predecessor's images.

        ⚠ B = 1 ONLY. `run_images` runs the tower on one frame set and writes
        row 0; a B > 1 policy builds its prefix from vision-cache rows with
        `build_prefix_from_segment`. Raised, not assumed, because rows 1.. of
        a stale `prefix_buf` are finite, plausible and wrong.
        """
        if Self.B != 1:
            raise Error(
                "SmolVLAPolicy.build_prefix: the image path is B = 1; this"
                " policy is B = " + String(Self.B)
                + " — use build_prefix_from_segment with vision-cache rows"
            )
        self.embed_images[target](images, ctx)
        self._tail_and_prefill[target](lang_ids, raw_state, ctx)

    def embed_images[
        target: StaticString
    ](mut self, mut images: Tensor, ctx: Optional[DeviceContext] = None) raises:
        """The image segment of the prefix, and nothing after it — SigLIP,
        shuffle, connector, sqrt(960) — into `prefix_buf[0 : IMG_SEG]`. What
        `vision_cache.mojo` stores; `image_segment` reads it back."""
        self.prefix.run_images[target, SMOLVLA_CONNECTOR_IN](
            self.vision, self.connector, images, self.prefix_buf, ctx
        )

    def image_segment[
        target: StaticString
    ](
        mut self, mut dst: List[Float32], ctx: Optional[DeviceContext] = None
    ) raises:
        """`dst` <- `prefix_buf[0 : IMG_SEG]`, after `embed_images`.

        ⚠ SYNCHRONISES on GPU (a download). Used when BUILDING the cache, once
        per store row, never in the training step."""
        comptime if target != "cpu":
            self.prefix_buf.download(ctx.value())
        dst.resize(Self.Prefix.IMG_SEG, 0.0)
        for i in range(Self.Prefix.IMG_SEG):
            dst[i] = Float32(self.prefix_buf.data[i])

    def build_prefix_from_segment[
        target: StaticString
    ](
        mut self,
        ref seg: List[Float32],
        ref lang_ids: List[Int],
        ref raw_state: List[Float32],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`build_prefix` with the image segments supplied instead of computed.

        `seg` is `B` vision-cache rows back to back (`B * IMG_SEG` floats),
        `raw_state` is `B` poses back to back, `lang_ids` one instruction or
        `B` (see `run_tail`). The prefix from here on is IDENTICAL to
        `build_prefix`'s: the same `run_tail`, the same cache reset, the same
        prefill. `test_vision_cache` asserts the prefix buffer and the prefill
        output are bit-identical between the two doors at B = 1;
        `test_train_step_batched` that B rows reproduce B separate runs.

        ⚠ The whole `prefix_buf` host slab is uploaded, language and state
        slots included; `run_tail` then overwrites those ON THE DEVICE, so
        whatever the host slab held there is never read.
        """
        comptime SEG = Self.Prefix.IMG_SEG
        comptime ROW = Self.P * Self.W
        if len(seg) != Self.B * SEG:
            raise Error(
                "SmolVLAPolicy.build_prefix_from_segment: got "
                + String(len(seg)) + " floats, B = " + String(Self.B)
                + " rows of " + String(SEG) + " need " + String(Self.B * SEG)
            )
        self.prefix_buf.ensure(Self.Prefix.OUT_N)
        for b in range(Self.B):
            for i in range(SEG):
                self.prefix_buf.data[b * ROW + i] = Scalar[DT](seg[b * SEG + i])
        comptime if target != "cpu":
            self.prefix_buf.upload_resident(ctx.value())
        self._tail_and_prefill[target](lang_ids, raw_state, ctx)

    def _tail_and_prefill[
        target: StaticString
    ](
        mut self,
        ref lang_ids: List[Int],
        ref raw_state: List[Float32],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Everything after the image segment: the state normalisation, the
        language and state tokens, the cache reset, the prefill. ONE copy,
        shared by both `build_prefix` doors."""
        if self.stats.state_dim() == 0:
            raise Error(
                "SmolVLAPolicy: no stats — call load_stats() before"
                " build_prefix, or the state goes in unnormalised"
            )

        # ── state: normalise, pad to 32, one row per batch element ───────
        var rdim = self.stats.state_dim()
        if len(raw_state) != Self.B * rdim:
            raise Error(
                "SmolVLAPolicy: raw_state has " + String(len(raw_state))
                + " values; B = " + String(Self.B) + " poses of "
                + String(rdim) + " need " + String(Self.B * rdim)
            )
        # ⚠ `ensure` on both paths: it sizes `data`, the host slab. The GPU
        # path differs only by the `upload` after the writes.
        self.state_buf.ensure(Self.B * Self.SDIM)
        var st = List[Float32]()
        var one = List[Float32]()
        for b in range(Self.B):
            one.clear()
            for j in range(rdim):
                one.append(raw_state[b * rdim + j])
            normalize_state(self.stats, one, st, Self.SDIM)
            for i in range(Self.SDIM):
                self.state_buf.data[b * Self.SDIM + i] = Scalar[DT](st[i])
        comptime if target != "cpu":
            # `upload_resident`: reuses the buffer instead of recreating it.
            self.state_buf.upload_resident(ctx.value())

        # ── language + state tokens, then prefill ────────────────────────
        self.prefix.run_tail[target, Self.VOCAB, Self.SDIM](
            self.embed.weight.val, self.state_proj, lang_ids, self.state_buf,
            self.prefix_buf, ctx,
        )
        self.cache.reset()
        self.prefill.run[target](
            self.tower, self.cache, self.prefix_buf, self.prefill_out, ctx
        )

    def select_action[
        target: StaticString
    ](
        mut self,
        mut images: Tensor,
        ref lang_ids: List[Int],
        ref raw_state: List[Float32],
        mut noise: Tensor,
        mut actions: List[Float32],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One observation to `[CHUNK, action_dim]` in robot units.

        `images` is `[N_CAM, 3*512*512]` in `[-1, 1]` (see
        `smolvla/observation.mojo`); `raw_state` is the robot's own joint
        values; `noise` is x_1, `[B, CHUNK*ADIM]`, supplied by the caller so the
        RNG stays outside and a gate can pin it.
        """
        self.start_action[target](images, lang_ids, raw_state, noise, ctx)
        comptime if target != "cpu":
            ctx.value().synchronize()
        self.finish_action[target](actions)

    def start_action[
        target: StaticString
    ](
        mut self,
        mut images: Tensor,
        ref lang_ids: List[Int],
        ref raw_state: List[Float32],
        mut noise: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """ENQUEUE one observation's query and return without waiting for it.

        ⚠ THE POINT IS THE CALLER'S OTHER WORK. On an Orin the query is ~660 ms
        of GPU time behind ~35 ms of launches, and a robot loop that blocks on
        it commands nothing for two thirds of a second — the arm freezes, then
        sprints to catch up when the chunk lands. Split, the control loop keeps
        commanding the CURRENT chunk while the next one is computed.

        Nothing between here and `finish_action` synchronises: the prefix, the
        prefill and the ten Euler steps are all enqueues, and the only D2H copy
        is staged (`download_enqueue`), not waited on.

        ⚠ `images`, `noise` and the policy's buffers ARE THE QUEUE'S INPUTS
        until the work completes. Overwriting them before `finish_action` — the
        next observation into the same `images` tensor, say — races the GPU.
        """
        self.build_prefix[target](images, lang_ids, raw_state, ctx)
        self.sampler.sample[target, Self.P](
            self.expert, self.cache, self.denoiser, self.action_in,
            self.time_mlp_in, self.time_mlp_out, self.action_out,
            noise, self.chunk_buf, ctx,
        )
        comptime if target != "cpu":
            self.chunk_buf.download_enqueue(ctx.value())

    def finish_action[
        target: StaticString
    ](mut self, mut actions: List[Float32]) raises:
        """Collect the chunk `start_action` enqueued, in robot units.

        ⚠ THE CALLER MUST HAVE SYNCHRONISED the context first. This reads the
        staged host buffer; without the sync it reads whatever was there, which
        on a second query is the PREVIOUS chunk — plausible actions, silently
        one observation stale."""
        comptime if target != "cpu":
            self.chunk_buf.download_finalize()

        # ── back to robot units, padded dims dropped ─────────────────────
        var flat = List[Float32]()
        for i in range(Self.B * Self.CHUNK * Self.ADIM):
            flat.append(Float32(self.chunk_buf.data[i]))
        unnormalize_action(self.stats, flat, Self.CHUNK, actions, Self.ADIM)
