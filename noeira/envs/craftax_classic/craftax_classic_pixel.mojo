"""Craftax-Classic — RGB sprite pixel observation (matches paper's spec).

Each output pixel maps to one block in the player-centered 9×7 view (top) or
to one cell in the inventory bar (bottom 2 rows). Blocks/mobs/player are
rendered from real PNG sprites pre-resized to `BLOCK_PIXEL_SIZE` (10) via
nearest-neighbor — the exact pipeline the original Craftax paper uses for
agent observations.

Layout:
    Obs canvas    : (VIEW_H + INVENTORY_OBS_HEIGHT) × VIEW_W × BPS²
                  = 9 × 9 × 100 pixels = 90 × 90 × 3 RGB
    OBS_DIM       : 3 * 90 * 90 = 24300 (channel-first, row-major)
    Inventory row : 0 → [HP, FD, DR, EN, sapling, wood, stone, coal, iron]
                    1 → [diamond, w-pick, s-pick, i-pick, w-sword, s-sword,
                         i-sword, ·, ·]

State: identical to `CraftaxClassicEnv` (STATE_SIZE = 4235).

GPU workspace:
    shared (one copy across the batch) : NUM_SPRITES × BPS² × 4 floats
                                       = 41 × 100 × 4 = 16400 floats
    per-env                            : 0 (single frame, no stack)
"""

from std.memory import alloc, unsafe_memset, dealloc
from noeira.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from noeira.nn.constants import DT as gpu_dtype
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom

from .craftax_classic import CraftaxClassicEnv, CraftaxState, CraftaxAction
from .constants import (
    MAP_H,
    MAP_W,
    NUM_ACTIONS,
    VIEW_H,
    VIEW_W,
    MAX_ZOMBIES,
    MAX_COWS,
    MAX_SKELETONS,
    MAX_ARROWS,
    MAX_PLANTS,
    MOB_FY,
    MOB_FX,
    MOB_HP,
    ARROW_FDIR,
    PLANT_FY,
    PLANT_FX,
    PLANT_FAGE,
    PLANT_RIPEN_AGE,
    BLOCK_OUT_OF_BOUNDS,
    INV_WOOD,
    INV_STONE,
    INV_COAL,
    INV_IRON,
    INV_DIAMOND,
    INV_SAPLING,
    INV_WOOD_PICKAXE,
    INV_STONE_PICKAXE,
    INV_IRON_PICKAXE,
    INV_WOOD_SWORD,
    INV_STONE_SWORD,
    INV_IRON_SWORD,
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
    DIR_DOWN,
)
from .state import (
    STATE_SIZE,
    S_MAP_BASE,
    S_PLAYER_POS,
    S_PLAYER_DIR,
    S_LIGHT_LEVEL,
    S_IS_SLEEPING,
    S_INV_BASE,
    S_INTRINSICS_BASE,
    s_zombie,
    s_cow,
    s_skeleton,
    s_arrow,
    s_plant,
    s_plant_mask,
)
from .game_logic import apply_step_inline
from .craftax_classic_sprites import (
    build_agent_atlas,
    agent_atlas_size,
    NUM_SPRITES,
    SPR_PLANT_YOUNG,
    SPR_PLANT_RIPE,
    SPR_ZOMBIE,
    SPR_COW,
    SPR_SKELETON,
    SPR_ARROW_UP,
    SPR_ARROW_DOWN,
    SPR_ARROW_LEFT,
    SPR_ARROW_RIGHT,
    SPR_PLAYER_UP,
    SPR_PLAYER_DOWN,
    SPR_PLAYER_LEFT,
    SPR_PLAYER_RIGHT,
    SPR_PLAYER_SLEEP,
    SPR_INV_WOOD,
    SPR_INV_SAPLING,
    SPR_INV_WOOD_PICKAXE,
    SPR_INV_STONE_PICKAXE,
    SPR_INV_IRON_PICKAXE,
    SPR_INV_WOOD_SWORD,
    SPR_INV_STONE_SWORD,
    SPR_INV_IRON_SWORD,
    SPR_ICON_HEALTH,
    SPR_ICON_FOOD,
    SPR_ICON_DRINK,
    SPR_ICON_ENERGY,
)


# ============================================================================
# Pixel obs geometry
# ============================================================================

comptime BLOCK_PIXEL_SIZE: Int = 10                # Craftax BLOCK_PIXEL_SIZE_AGENT
comptime INVENTORY_OBS_HEIGHT: Int = 2             # rows of inventory cells
comptime OBS_PIX_H: Int = (VIEW_H + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE  # 90
comptime OBS_PIX_W: Int = VIEW_W * BLOCK_PIXEL_SIZE                          # 90
comptime OBS_CHANNELS: Int = 3
comptime PIXEL_OBS_DIM: Int = OBS_CHANNELS * OBS_PIX_H * OBS_PIX_W            # 24300

comptime VIEW_PIX_H: Int = VIEW_H * BLOCK_PIXEL_SIZE                          # 70
comptime ATLAS_FLOATS: Int = NUM_SPRITES * BLOCK_PIXEL_SIZE * BLOCK_PIXEL_SIZE * 4

# Asset directory — relative to project root.
comptime ASSET_DIR: String = "noeira/envs/craftax_classic/assets"


# ============================================================================
# Inventory-cell → (sprite, value) mapping
# ============================================================================
#
# Inventory canvas is 2 rows × 9 cols of cells. Layout (row, col) → entry,
# matching `craftax_classic/renderer.py`:
#   (0,0) HP   (0,1) FD   (0,2) DR   (0,3) EN
#   (0,4) sapling   (0,5) wood   (0,6) stone   (0,7) coal   (0,8) iron
#   (1,0) diamond
#   (1,1) w-pick    (1,2) s-pick    (1,3) i-pick
#   (1,4) w-sword   (1,5) s-sword   (1,6) i-sword
#   (1,7), (1,8): empty
#
# Returns (sprite_idx, value); sprite_idx == -1 means "empty cell".

@always_inline
def _inv_cell_sprite_and_value(
    state: Pointer[Float32, MutAnyOrigin],
    row: Int,
    col: Int,
) -> Tuple[Int, Int]:
    if row == 0:
        if col == 0:
            return (SPR_ICON_HEALTH, Int(state[unsafe_offset=S_INTRINSICS_BASE + INTRINSIC_HEALTH]))
        elif col == 1:
            return (SPR_ICON_FOOD, Int(state[unsafe_offset=S_INTRINSICS_BASE + INTRINSIC_FOOD]))
        elif col == 2:
            return (SPR_ICON_DRINK, Int(state[unsafe_offset=S_INTRINSICS_BASE + INTRINSIC_DRINK]))
        elif col == 3:
            return (SPR_ICON_ENERGY, Int(state[unsafe_offset=S_INTRINSICS_BASE + INTRINSIC_ENERGY]))
        elif col == 4:
            return (SPR_INV_SAPLING, Int(state[unsafe_offset=S_INV_BASE + INV_SAPLING]))
        elif col == 5:
            return (SPR_INV_WOOD, Int(state[unsafe_offset=S_INV_BASE + INV_WOOD]))
        elif col == 6:
            return (4, Int(state[unsafe_offset=S_INV_BASE + INV_STONE]))   # SPR_STONE
        elif col == 7:
            return (8, Int(state[unsafe_offset=S_INV_BASE + INV_COAL]))    # SPR_COAL
        elif col == 8:
            return (9, Int(state[unsafe_offset=S_INV_BASE + INV_IRON]))    # SPR_IRON
    else:  # row == 1
        if col == 0:
            return (10, Int(state[unsafe_offset=S_INV_BASE + INV_DIAMOND]))  # SPR_DIAMOND
        elif col == 1:
            return (SPR_INV_WOOD_PICKAXE, Int(state[unsafe_offset=S_INV_BASE + INV_WOOD_PICKAXE]))
        elif col == 2:
            return (SPR_INV_STONE_PICKAXE, Int(state[unsafe_offset=S_INV_BASE + INV_STONE_PICKAXE]))
        elif col == 3:
            return (SPR_INV_IRON_PICKAXE, Int(state[unsafe_offset=S_INV_BASE + INV_IRON_PICKAXE]))
        elif col == 4:
            return (SPR_INV_WOOD_SWORD, Int(state[unsafe_offset=S_INV_BASE + INV_WOOD_SWORD]))
        elif col == 5:
            return (SPR_INV_STONE_SWORD, Int(state[unsafe_offset=S_INV_BASE + INV_STONE_SWORD]))
        elif col == 6:
            return (SPR_INV_IRON_SWORD, Int(state[unsafe_offset=S_INV_BASE + INV_IRON_SWORD]))
    return (-1, 0)


# ============================================================================
# Per-pixel render — shared by CPU & GPU paths
# ============================================================================
#
# Output is channel-first `[C, H, W]` row-major. For a given thread that owns
# output pixel `(c, h, w)`, we compute the RGB of `(h, w)` once and pick
# channel `c` at the end. With one thread per (c, h, w) we render each
# pixel three times — wasteful for 3× the work, but the indexing is the
# simplest possible and the kernel is still cheap.
#
# Alternative: one thread per `(h, w)`, write 3 floats. We use that here.


@always_inline
def _atlas_sample(
    atlas: Pointer[Float32, MutAnyOrigin],
    sprite_idx: Int,
    ly: Int,
    lx: Int,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Read RGBA from atlas slot at local pixel (ly, lx)."""
    var off = (
        sprite_idx * BLOCK_PIXEL_SIZE * BLOCK_PIXEL_SIZE * 4
        + (ly * BLOCK_PIXEL_SIZE + lx) * 4
    )
    return (atlas[unsafe_offset=off + 0], atlas[unsafe_offset=off + 1], atlas[unsafe_offset=off + 2], atlas[unsafe_offset=off + 3])


@always_inline
def _composite(
    base_r: Float32, base_g: Float32, base_b: Float32,
    over_r: Float32, over_g: Float32, over_b: Float32, over_a: Float32,
) -> Tuple[Float32, Float32, Float32]:
    """Alpha-composite `over` onto `base` (over is premultiplied form)."""
    var inv = Float32(1.0) - over_a
    return (
        base_r * inv + over_r * over_a,
        base_g * inv + over_g * over_a,
        base_b * inv + over_b * over_a,
    )


@always_inline
def _player_sprite_for(dir_: Int, sleeping: Bool) -> Int:
    if sleeping:
        return SPR_PLAYER_SLEEP
    if dir_ == DIR_LEFT:
        return SPR_PLAYER_LEFT
    elif dir_ == DIR_RIGHT:
        return SPR_PLAYER_RIGHT
    elif dir_ == DIR_UP:
        return SPR_PLAYER_UP
    else:
        return SPR_PLAYER_DOWN


@always_inline
def _arrow_sprite_for(dir_: Int) -> Int:
    if dir_ == DIR_LEFT:
        return SPR_ARROW_LEFT
    elif dir_ == DIR_RIGHT:
        return SPR_ARROW_RIGHT
    elif dir_ == DIR_UP:
        return SPR_ARROW_UP
    else:
        return SPR_ARROW_DOWN


@always_inline
def _render_pixel_rgb(
    state: Pointer[Float32, MutAnyOrigin],
    atlas: Pointer[Float32, MutAnyOrigin],
    h: Int,
    w: Int,
) -> Tuple[Float32, Float32, Float32]:
    """Return RGB for output pixel (h, w) ∈ [0, 90)²."""
    var py = Int(state[unsafe_offset=S_PLAYER_POS])
    var px = Int(state[unsafe_offset=S_PLAYER_POS + 1])

    if h < VIEW_PIX_H:
        # ---- Game view ----
        var vy = h // BLOCK_PIXEL_SIZE  # 0..6
        var vx = w // BLOCK_PIXEL_SIZE  # 0..8
        var ly = h % BLOCK_PIXEL_SIZE
        var lx = w % BLOCK_PIXEL_SIZE
        var wy = py - VIEW_H // 2 + vy
        var wx = px - VIEW_W // 2 + vx
        var in_bounds = (
            wy >= 0 and wy < MAP_H and wx >= 0 and wx < MAP_W
        )

        var block_id = BLOCK_OUT_OF_BOUNDS
        if in_bounds:
            block_id = Int(state[unsafe_offset=S_MAP_BASE + wy * MAP_W + wx])

        # Background = block sprite (always opaque for blocks).
        var base = _atlas_sample(atlas, block_id, ly, lx)
        var r = base[0]
        var g = base[1]
        var b = base[2]

        if in_bounds:
            # Plants (alpha-blended overlay).
            for i in range(MAX_PLANTS):
                if state[unsafe_offset=s_plant_mask(i)] < Float32(0.5):
                    continue
                if (
                    Int(state[unsafe_offset=s_plant(i, PLANT_FY)]) == wy
                    and Int(state[unsafe_offset=s_plant(i, PLANT_FX)]) == wx
                ):
                    var age = Int(state[unsafe_offset=s_plant(i, PLANT_FAGE)])
                    var ps = SPR_PLANT_RIPE if age >= PLANT_RIPEN_AGE else SPR_PLANT_YOUNG
                    var p = _atlas_sample(atlas, ps, ly, lx)
                    var c = _composite(r, g, b, p[0], p[1], p[2], p[3])
                    r = c[0]; g = c[1]; b = c[2]

            # Cows, zombies, skeletons, arrows.
            for i in range(MAX_COWS):
                if state[unsafe_offset=s_cow(i, MOB_HP)] > Float32(0.0):
                    if (
                        Int(state[unsafe_offset=s_cow(i, MOB_FY)]) == wy
                        and Int(state[unsafe_offset=s_cow(i, MOB_FX)]) == wx
                    ):
                        var p = _atlas_sample(atlas, SPR_COW, ly, lx)
                        var c = _composite(r, g, b, p[0], p[1], p[2], p[3])
                        r = c[0]; g = c[1]; b = c[2]
            for i in range(MAX_ZOMBIES):
                if state[unsafe_offset=s_zombie(i, MOB_HP)] > Float32(0.0):
                    if (
                        Int(state[unsafe_offset=s_zombie(i, MOB_FY)]) == wy
                        and Int(state[unsafe_offset=s_zombie(i, MOB_FX)]) == wx
                    ):
                        var p = _atlas_sample(atlas, SPR_ZOMBIE, ly, lx)
                        var c = _composite(r, g, b, p[0], p[1], p[2], p[3])
                        r = c[0]; g = c[1]; b = c[2]
            for i in range(MAX_SKELETONS):
                if state[unsafe_offset=s_skeleton(i, MOB_HP)] > Float32(0.0):
                    if (
                        Int(state[unsafe_offset=s_skeleton(i, MOB_FY)]) == wy
                        and Int(state[unsafe_offset=s_skeleton(i, MOB_FX)]) == wx
                    ):
                        var p = _atlas_sample(atlas, SPR_SKELETON, ly, lx)
                        var c = _composite(r, g, b, p[0], p[1], p[2], p[3])
                        r = c[0]; g = c[1]; b = c[2]
            for i in range(MAX_ARROWS):
                if state[unsafe_offset=s_arrow(i, MOB_HP)] > Float32(0.0):
                    if (
                        Int(state[unsafe_offset=s_arrow(i, MOB_FY)]) == wy
                        and Int(state[unsafe_offset=s_arrow(i, MOB_FX)]) == wx
                    ):
                        var ad = Int(state[unsafe_offset=s_arrow(i, ARROW_FDIR)])
                        var p = _atlas_sample(
                            atlas, _arrow_sprite_for(ad), ly, lx
                        )
                        var c = _composite(r, g, b, p[0], p[1], p[2], p[3])
                        r = c[0]; g = c[1]; b = c[2]

            # Player at view center.
            if wy == py and wx == px:
                var pdir = Int(state[unsafe_offset=S_PLAYER_DIR])
                var sleeping = state[unsafe_offset=S_IS_SLEEPING] > Float32(0.5)
                var ps_idx = _player_sprite_for(pdir, sleeping)
                var p = _atlas_sample(atlas, ps_idx, ly, lx)
                var c = _composite(r, g, b, p[0], p[1], p[2], p[3])
                r = c[0]; g = c[1]; b = c[2]

        # Day/night dim, clamped so the agent always sees something.
        var light = state[unsafe_offset=S_LIGHT_LEVEL]
        if light < Float32(0.3):
            light = Float32(0.3)
        if light > Float32(1.0):
            light = Float32(1.0)
        return (r * light, g * light, b * light)
    else:
        # ---- Inventory bar ----
        var iy = h - VIEW_PIX_H  # 0..19
        var row = iy // BLOCK_PIXEL_SIZE  # 0 or 1
        var col = w // BLOCK_PIXEL_SIZE   # 0..8
        var ly = iy % BLOCK_PIXEL_SIZE
        var lx = w % BLOCK_PIXEL_SIZE

        var sv = _inv_cell_sprite_and_value(state, row, col)
        var sprite = sv[0]
        var value = sv[1]
        # Empty cell or zero-count slot → dim background.
        if sprite < 0 or value <= 0:
            return (Float32(0.05), Float32(0.05), Float32(0.08))
        var p = _atlas_sample(atlas, sprite, ly, lx)
        # Cells with no alpha get the dim background showing through.
        var bg_r = Float32(0.08)
        var bg_g = Float32(0.08)
        var bg_b = Float32(0.10)
        var c = _composite(bg_r, bg_g, bg_b, p[0], p[1], p[2], p[3])
        return (c[0], c[1], c[2])


# ============================================================================
# CraftaxClassicPixelEnv
# ============================================================================


struct CraftaxClassicPixelEnv[DTYPE: DType = DType.float32](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Craftax-Classic with 3×90×90 RGB sprite-based pixel obs.

    Channel-first layout (C, H, W) flat row-major. Single frame, no stack —
    same as the published Craftax baselines (`render_craftax_pixels` /
    `BLOCK_PIXEL_SIZE_AGENT`).

    Physics + rendering for human play delegate to `CraftaxClassicEnv`.
    """

    comptime dtype = Self.DTYPE
    comptime StateType = CraftaxState
    comptime ActionType = CraftaxAction

    comptime STATE_SIZE: Int = STATE_SIZE
    comptime OBS_DIM: Int = PIXEL_OBS_DIM
    comptime NUM_ACTIONS: Int = NUM_ACTIONS
    comptime STEP_WS_SHARED: Int = ATLAS_FLOATS
    comptime STEP_WS_PER_ENV: Int = 0

    var inner: CraftaxClassicEnv[Self.DTYPE]
    var _atlas: Pointer[Float32, MutUntrackedOrigin]   # CPU-side atlas
    var _atlas_loaded: Bool

    def __init__(out self):
        self.inner = CraftaxClassicEnv[Self.DTYPE]()
        # Try to load the CPU atlas. Falls back silently to a black-screen
        # atlas if PIL/assets are unavailable (the env will still step).
        self._atlas_loaded = False
        try:
            self._atlas = build_agent_atlas(ASSET_DIR, BLOCK_PIXEL_SIZE)
            self._atlas_loaded = True
        except e:
            print("Craftax pixel env: atlas load failed (", String(e), ")")
            self._atlas = alloc[Float32]({count = ATLAS_FLOATS}).unsafe_leak()
            for i in range(ATLAS_FLOATS):
                self._atlas[unsafe_offset=i] = Float32(0.0)

    def __deinit__(deinit self):
        if Int(self._atlas) != 0:
            self._atlas.unsafe_free()

    # ========================================================================
    # CPU: render current state into obs buffer
    # ========================================================================

    @always_inline
    def _render_current(
        self, mut obs: Pointer[Scalar[Self.DTYPE], MutAnyOrigin]
    ):
        var state_ptr = rebind[Pointer[Float32, MutAnyOrigin]](
            self.inner.state.unsafe_ptr().unsafe_bitcast[Float32]()
        )
        var atlas = self._atlas
        # Channel-first: obs[c, h, w] at offset c * (H*W) + h*W + w.
        comptime HW = OBS_PIX_H * OBS_PIX_W
        for h in range(OBS_PIX_H):
            for w in range(OBS_PIX_W):
                var rgb = _render_pixel_rgb(
                    state_ptr, atlas.as_unsafe_any_origin(), h, w
                )
                obs[unsafe_offset=0 * HW + h * OBS_PIX_W + w] = Scalar[Self.DTYPE](rgb[0])
                obs[unsafe_offset=1 * HW + h * OBS_PIX_W + w] = Scalar[Self.DTYPE](rgb[1])
                obs[unsafe_offset=2 * HW + h * OBS_PIX_W + w] = Scalar[Self.DTYPE](rgb[2])

    # ========================================================================
    # Env trait
    # ========================================================================

    def reset(mut self) -> CraftaxState:
        return self.inner.reset()

    def reset_with_seed(
        mut self, seed: UInt64, always_diamond: Bool = False
    ) -> CraftaxState:
        return self.inner.reset_with_seed(seed, always_diamond)

    def step(
        mut self, action: CraftaxAction, verbose: Bool = False
    ) -> Tuple[CraftaxState, Scalar[Self.DTYPE], Bool]:
        return self.inner.step(action, verbose)

    def get_state(mut self) -> CraftaxState:
        return self.inner.get_state()

    def close(mut self):
        self.inner.close()

    def action_from_index(self, action_idx: Int) -> CraftaxAction:
        return self.inner.action_from_index(action_idx)

    def num_actions(self) -> Int:
        return NUM_ACTIONS

    def obs_dim(self) -> Int:
        return PIXEL_OBS_DIM

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: CraftaxState) -> Int:
        return state.index

    # ========================================================================
    # BoxDiscreteActionEnv
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        var obs_arr_a = alloc[Scalar[Self.DTYPE]]({count = PIXEL_OBS_DIM})
        var obs_arr = obs_arr_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var obs_ptr = rebind[Pointer[Scalar[Self.DTYPE], MutAnyOrigin]](
            obs_arr
        )
        self._render_current(obs_ptr)
        var obs = List[Scalar[Self.DTYPE]](capacity=PIXEL_OBS_DIM)
        for i in range(PIXEL_OBS_DIM):
            obs.append(obs_arr[unsafe_offset=i])
        dealloc(obs_arr_a^)
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.DTYPE]]:
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        var result = self.inner._step_impl(action)
        return (self.get_obs_list(), result[0], result[1])

    # ========================================================================
    # RenderableEnv — delegate to inner (human playable window)
    # ========================================================================

    def init_renderer(mut self) raises -> Bool:
        return self.inner.init_renderer()

    def render_frame(mut self) raises -> None:
        self.inner.render_frame()

    def close_renderer(mut self) raises -> None:
        self.inner.close_renderer()

    def is_renderer_open(self) -> Bool:
        return self.inner.is_renderer_open()

    def check_renderer_quit(mut self) -> Bool:
        return self.inner.check_renderer_quit()

    def renderer_delay(self, ms: Int) -> None:
        self.inner.renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

    # ========================================================================
    # GPU kernels
    # ========================================================================

    comptime TPB: Int = 256

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        CraftaxClassicEnv[Self.DTYPE].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, rng_seed=rng_seed
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
        workspace_ptr: Optional[
            Pointer[Scalar[gpu_dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        CraftaxClassicEnv[Self.DTYPE].selective_reset_kernel_gpu[
            BATCH_SIZE, STATE_SIZE
        ](
            ctx,
            states_buf,
            dones_buf,
            rng_seed=rng_seed,
            workspace_ptr=workspace_ptr,
            rng_counter_ptr=rng_counter_ptr,
        )

    def init_step_workspace_gpu_with_atlas[
        BATCH_SIZE: Int,
    ](
        self,
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Copy the CPU atlas into the shared region of the GPU workspace.

        Called by the training loop before the first step. The atlas is
        constant for the run; per-env workspace size is zero so nothing
        else needs initializing.
        """
        # Copy host atlas into the first ATLAS_FLOATS of workspace_buf.
        # workspace layout: [shared (ATLAS_FLOATS) | per-env padding (0)]
        ctx.enqueue_copy(workspace_buf, self._atlas)

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype],) raises:
        """Static fallback that loads + uploads the atlas in-place.

        Built for trait conformance; the training loop ends up calling
        whichever overload is available. The non-static `_with_atlas`
        version is preferred because it reuses the CPU-side atlas already
        held by the env instance.
        """
        var host = build_agent_atlas(ASSET_DIR, BLOCK_PIXEL_SIZE)
        ctx.enqueue_copy(workspace_buf, host)
        ctx.synchronize()
        host.unsafe_free()

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        pass

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Cold extract used after reset. Without a workspace pointer here,
        we re-upload the atlas as a one-shot device buffer. Training paths
        always have the workspace available, so this is the eval/debug
        fallback only.
        """
        # Allocate a temporary atlas buffer and populate it.
        var atlas_buf = ctx.enqueue_create_buffer[gpu_dtype](ATLAS_FLOATS)
        var host = build_agent_atlas(ASSET_DIR, BLOCK_PIXEL_SIZE)
        ctx.enqueue_copy(atlas_buf, host)
        ctx.synchronize()
        host.unsafe_free()

        Self._render_kernel[BATCH_SIZE, STATE_SIZE](
            ctx,
            states_buf,
            # `atlas_buf` is bound by `var` here but the origin cast lands on
            # the immutable side; the kernel ABI declares `MutAnyOrigin`.
            atlas_buf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True](),
            obs_buf,
        )

    @staticmethod
    def _render_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[gpu_dtype],
        atlas_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """One thread per output pixel. Writes 3 channels per pixel."""
        comptime PIX_TOTAL = BATCH_SIZE * OBS_PIX_H * OBS_PIX_W
        comptime PIX_BLOCKS = (PIX_TOTAL + Self.TPB - 1) // Self.TPB
        # `states_buf` is borrowed immutably, so its pointer now carries an
        # immutable origin; the kernel ABI declares `MutAnyOrigin`. Device
        # allocations are outside Mojo's origin tracking, so this restores the
        # pre-nightly typing without granting the kernel any new access.
        var states_ptr = (
            states_buf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
        )
        var obs_ptr = obs_buf.unsafe_ptr()

        @always_inline
        def render_wrapper(
            states_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
            atlas_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
            obs_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= PIX_TOTAL:
                return
            comptime HW = OBS_PIX_H * OBS_PIX_W
            var env_idx = tid // HW
            var pix = tid % HW
            var h = pix // OBS_PIX_W
            var w = pix % OBS_PIX_W

            var state = states_ptr.unsafe_offset(env_idx * STATE_SIZE)
            var rgb = _render_pixel_rgb(state, atlas_ptr, h, w)
            var env_obs = obs_ptr.unsafe_offset(env_idx * PIXEL_OBS_DIM)
            env_obs[unsafe_offset=0 * HW + pix] = rgb[0]
            env_obs[unsafe_offset=1 * HW + pix] = rgb[1]
            env_obs[unsafe_offset=2 * HW + pix] = rgb[2]

        ctx.enqueue_function[render_wrapper](
            states_ptr,
            atlas_ptr,
            obs_ptr,
            grid_dim=(PIX_BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut terminated_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        workspace_ptr: Optional[
            Pointer[Scalar[gpu_dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """Full step: physics → render into obs."""
        var states_ptr = states_buf.unsafe_ptr()
        # Borrowed immutably; the kernel ABI wants `MutAnyOrigin`. Device
        # allocations are outside origin tracking, so this only restores the
        # pre-nightly typing.
        var actions_ptr = (
            actions_buf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
        )
        var rewards_ptr = rewards_buf.unsafe_ptr()
        var dones_ptr = dones_buf.unsafe_ptr()
        var terminated_ptr = terminated_buf.unsafe_ptr()
        var obs_ptr = obs_buf.unsafe_ptr()
        var ws_ptr = workspace_ptr.value()

        # ── Kernel 1: Physics ─────────────────────────────────────────────
        comptime PHYS_BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed_s = Scalar[DType.uint64](rng_seed)

        @always_inline
        def physics_wrapper(
            states_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
            actions_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
            rewards_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
            dones_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
            terminated_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return
            var state = states_ptr.unsafe_offset(e * STATE_SIZE)
            var action = Int(actions_ptr[unsafe_offset=e])
            var per_env_seed = (
                UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(e) + UInt64(1)
            )
            var rng = PhiloxRandom(seed=per_env_seed, offset=0)
            var r_done = apply_step_inline(state, action, rng)
            rewards_ptr[unsafe_offset=e] = Scalar[gpu_dtype](r_done[0])
            if r_done[1]:
                dones_ptr[unsafe_offset=e] = Scalar[gpu_dtype](1.0)
            else:
                dones_ptr[unsafe_offset=e] = Scalar[gpu_dtype](0.0)
            terminated_ptr[unsafe_offset=e] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[physics_wrapper](
            states_ptr,
            actions_ptr,
            rewards_ptr,
            dones_ptr,
            terminated_ptr,
            seed_s,
            grid_dim=(PHYS_BLOCKS,),
            block_dim=(Self.TPB,),
        )

        # ── Kernel 2: Render ──────────────────────────────────────────────
        # Atlas lives in shared region of workspace at offset 0.
        Self._render_kernel[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, ws_ptr, obs_buf
        )
