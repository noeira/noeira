"""Craftax-Classic environment — Phase 1 skeleton.

This file currently provides:
  - Trait-conforming `CraftaxClassicEnv` struct (CPU + GPU paths)
  - No-op reset / step that wires up state, obs, reward, done shapes correctly
  - Zero-filled GPU kernels that exercise the launch path

Game logic, world gen, and observation extraction land in later phases.
See `docs/CRAFTAX_PORT.md`.
"""

from std.memory import dealloc, alloc
from std.ffi import c_int, c_float
from noeira.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from noeira.nn.constants import DT as gpu_dtype
from noeira.render import Renderer2D, SDL_Color
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from .constants import (
    MAP_H,
    MAP_W,
    MAP_SIZE,
    VIEW_H,
    VIEW_W,
    MAX_TIMESTEPS,
    NUM_ACTIONS,
    NUM_ACHIEVEMENTS,
    NUM_INVENTORY,
    OBS_DIM,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
    DIR_DOWN,
    DAY_LENGTH,
    PLAYER_MAX_HEALTH,
    INTRINSIC_MAX,
    NUM_INTRINSICS,
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
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
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_STONE,
    BLOCK_TREE,
    BLOCK_WOOD,
    BLOCK_PATH,
    BLOCK_COAL,
    BLOCK_IRON,
    BLOCK_DIAMOND,
    BLOCK_CRAFTING_TABLE,
    BLOCK_FURNACE,
    BLOCK_SAND,
    BLOCK_LAVA,
    BLOCK_PLANT,
    BLOCK_RIPE_PLANT,
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
)
from .state import (
    STATE_SIZE,
    S_MAP_BASE,
    S_PLAYER_POS,
    S_PLAYER_DIR,
    S_INTRINSICS_BASE,
    S_INV_BASE,
    S_ZOMBIES_BASE,
    S_COWS_BASE,
    S_SKELETONS_BASE,
    S_ARROWS_BASE,
    S_PLANTS_BASE,
    S_PLANT_MASK_BASE,
    S_ACHIEVEMENTS_BASE,
    S_LIGHT_LEVEL,
    S_IS_SLEEPING,
    S_TIMESTEP,
    s_map,
    s_zombie,
    s_cow,
    s_skeleton,
    s_arrow,
    s_plant,
    s_plant_mask,
)
from .world_gen import (
    generate_world_cpu,
    generate_world_inline,
    calculate_light_level,
)
from .game_logic import apply_step_inline, extract_obs_inline
from .craftax_classic_sprites import (
    build_sprite_sheet,
    SPRITE_SIZE,
    SHEET_WIDTH as SPRITE_SHEET_WIDTH,
    SHEET_HEIGHT as SPRITE_SHEET_HEIGHT,
    SPRITE_BPP,
    SPR_OOB,
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
from std.random.philox import Random as PhiloxRandom
from noeira.core.fmt import fit
from noeira.core.tiles import Tile, ImmTile, tile


# Per-env GPU world-gen workspace: 4 noise fields × MAP_SIZE floats.
comptime WORLD_GEN_WS_PER_ENV: Int = 4 * MAP_SIZE


# ============================================================================
# State / Action wrapper types (used for trait conformance only)
# ============================================================================


@fieldwise_init
struct CraftaxState(Copyable, ImplicitlyCopyable, Movable, State):
    """Trait-required handle; the actual state lives in the env's flat array."""

    var index: Int

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit move: Self):
        self.index = move.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct CraftaxAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Discrete action wrapper (0..NUM_ACTIONS-1)."""

    var value: Int

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    def __init__(out self, *, deinit move: Self):
        self.value = move.value


# ============================================================================
# CraftaxClassicEnv
# ============================================================================


struct CraftaxClassicEnv[DTYPE: DType = DType.float32](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Craftax-Classic environment.

    Phase 1: skeleton only. `reset` zero-fills state, `step` increments the
    timestep and returns zero reward; done fires at MAX_TIMESTEPS. This is the
    minimum that exercises trait conformance and the GPU launch path.
    """

    # Trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = CraftaxState
    comptime ActionType = CraftaxAction

    # GPUDiscreteEnv constants
    comptime STATE_SIZE: Int = STATE_SIZE
    comptime OBS_DIM: Int = OBS_DIM
    comptime NUM_ACTIONS: Int = NUM_ACTIONS
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    # CPU state — flat array indexed by offsets from state.mojo
    var state: Array[Scalar[Self.dtype], STATE_SIZE]
    var done: Bool
    var _rng_counter: UInt64

    # Renderer (allocated lazily by init_renderer)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool
    var _sprite_pixels: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    var _has_sprites: Bool

    def __init__(out self):
        self.state = Array[Scalar[Self.dtype], STATE_SIZE](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False
        self._rng_counter = 0
        self._renderer = None
        self._renderer_initialized = False
        self._sprite_pixels = None
        self._has_sprites = False

    # ========================================================================
    # CPU: reset + step
    # ========================================================================

    def reset(mut self) -> CraftaxState:
        """Reset env with a freshly generated world.

        Uses an internal counter as seed; call `reset_with_seed` for a
        specific reproducible map.
        """
        self._rng_counter += 1
        return self.reset_with_seed(self._rng_counter)

    def reset_with_seed(
        mut self, seed: UInt64, always_diamond: Bool = False
    ) -> CraftaxState:
        # Zero everything first.
        for i in range(STATE_SIZE):
            self.state[i] = Scalar[Self.dtype](0.0)

        # Generate world directly into the map section of state.
        # `generate_world_cpu` writes block IDs as Float32; cast to dtype.
        var map_ptr = self.state.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_offset(S_MAP_BASE)
        var spawn = generate_world_cpu(
            seed, map_ptr.as_unsafe_any_origin(), always_diamond
        )
        var py = spawn[0]
        var px = spawn[1]

        self.state[S_PLAYER_POS] = Scalar[Self.dtype](py)
        self.state[S_PLAYER_POS + 1] = Scalar[Self.dtype](px)
        self.state[S_PLAYER_DIR] = Scalar[Self.dtype](DIR_UP)

        # Intrinsics: health/food/drink/energy all at max (9).
        for k in range(NUM_INTRINSICS):
            self.state[S_INTRINSICS_BASE + k] = Scalar[Self.dtype](
                INTRINSIC_MAX
            )

        self.state[S_LIGHT_LEVEL] = Scalar[Self.dtype](
            calculate_light_level(0, DAY_LENGTH)
        )
        self.state[S_IS_SLEEPING] = Scalar[Self.dtype](0.0)
        self.state[S_TIMESTEP] = Scalar[Self.dtype](0.0)

        self.done = False
        return CraftaxState(index=0)

    def step(
        mut self, action: CraftaxAction, verbose: Bool = False
    ) -> Tuple[CraftaxState, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            CraftaxState(index=Int(self.state[S_TIMESTEP])),
            result[0],
            result[1],
        )

    def _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Full Phase-3A step: crafting, do, place, move, plants, intrinsics."""
        self._rng_counter += 1
        var rng = PhiloxRandom(seed=self._rng_counter, offset=0)
        var state_ptr = self.state.unsafe_ptr().unsafe_bitcast[Float32]()
        var result = apply_step_inline(
            state_ptr.as_unsafe_any_origin(), action, rng
        )
        self.done = result[1]
        return (Scalar[Self.dtype](result[0]), self.done)

    # ========================================================================
    # Env trait methods
    # ========================================================================

    def get_state(mut self) -> CraftaxState:
        return CraftaxState(index=Int(self.state[S_TIMESTEP]))

    def close(mut self):
        pass

    def action_from_index(self, action_idx: Int) -> CraftaxAction:
        return CraftaxAction(value=action_idx)

    def num_actions(self) -> Int:
        return NUM_ACTIONS

    def obs_dim(self) -> Int:
        return OBS_DIM

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: CraftaxState) -> Int:
        return state.index

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv (CPU)
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs_arr = Array[Float32, OBS_DIM](fill=Float32(0.0))
        var obs_ptr = rebind[Pointer[Float32, MutAnyOrigin]](
            obs_arr.unsafe_ptr().unsafe_bitcast[Float32]()
        )
        var state_ptr = rebind[Pointer[Float32, MutAnyOrigin]](
            self.state.unsafe_ptr().unsafe_bitcast[Float32]()
        )
        extract_obs_inline(state_ptr, obs_ptr)
        var obs = List[Scalar[Self.dtype]](capacity=OBS_DIM)
        for i in range(OBS_DIM):
            obs.append(Scalar[Self.dtype](obs_arr[i]))
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action)
        return (self.get_obs_list(), result[0], result[1])

    # ========================================================================
    # GPU kernels (Phase 1: zero-fill / increment, exercise launch path only)
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
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)

        # Per-env scratch (4 noise fields). Lives only for this kernel call.
        var scratch_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * WORLD_GEN_WS_PER_ENV
        )
        var scratch = tile[BATCH_SIZE * WORLD_GEN_WS_PER_ENV](scratch_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed_scalar = Scalar[DType.uint64](rng_seed)

        @always_inline
        def reset_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            scratch: Tile[gpu_dtype, BATCH_SIZE * WORLD_GEN_WS_PER_ENV],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return

            # Zero this env's state slice (mobs, plants, achievements, etc.).
            for s in range(STATE_SIZE):
                states[e, s] = Scalar[gpu_dtype](0.0)

            # Slice per-env scratch and compute pointers.
            var ws_base = e * WORLD_GEN_WS_PER_ENV
            var scratch_ptr = scratch.ptr.unsafe_offset(ws_base)
            var water_ptr = scratch_ptr
            var mountain_ptr = scratch_ptr.unsafe_offset(MAP_SIZE)
            var path_ptr = scratch_ptr.unsafe_offset(2 * MAP_SIZE)
            var tree_ptr = scratch_ptr.unsafe_offset(3 * MAP_SIZE)
            var map_ptr = (
                states.ptr.unsafe_offset(e * STATE_SIZE + S_MAP_BASE)
            )

            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(
                e
            ) + UInt64(1)
            var spawn = generate_world_inline(
                per_env_seed,
                water_ptr,
                mountain_ptr,
                path_ptr,
                tree_ptr,
                map_ptr,
                False,
            )
            states[e, S_PLAYER_POS] = Scalar[gpu_dtype](spawn[0])
            states[e, S_PLAYER_POS + 1] = Scalar[gpu_dtype](spawn[1])
            states[e, S_PLAYER_DIR] = Scalar[gpu_dtype](DIR_UP)
            for k in range(NUM_INTRINSICS):
                states[e, S_INTRINSICS_BASE + k] = Scalar[gpu_dtype](
                    INTRINSIC_MAX
                )
            states[e, S_LIGHT_LEVEL] = Scalar[gpu_dtype](0.7969252)
            states[e, S_IS_SLEEPING] = Scalar[gpu_dtype](0.0)
            states[e, S_TIMESTEP] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[reset_wrapper](
            states,
            scratch,
            seed_scalar,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
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
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var dones = tile[BATCH_SIZE](dones_buf)

        var scratch_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * WORLD_GEN_WS_PER_ENV
        )
        var scratch = tile[BATCH_SIZE * WORLD_GEN_WS_PER_ENV](scratch_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed_scalar = Scalar[DType.uint64](rng_seed)

        @always_inline
        def selective_reset_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            dones: Tile[gpu_dtype, BATCH_SIZE],
            scratch: Tile[gpu_dtype, BATCH_SIZE * WORLD_GEN_WS_PER_ENV],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return
            if dones[e] <= Scalar[gpu_dtype](0.5):
                return

            for s in range(STATE_SIZE):
                states[e, s] = Scalar[gpu_dtype](0.0)

            var ws_base = e * WORLD_GEN_WS_PER_ENV
            var scratch_ptr = scratch.ptr.unsafe_offset(ws_base)
            var map_ptr = (
                states.ptr.unsafe_offset(e * STATE_SIZE + S_MAP_BASE)
            )
            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(
                e
            ) + UInt64(1)
            var spawn = generate_world_inline(
                per_env_seed,
                scratch_ptr,
                scratch_ptr.unsafe_offset(MAP_SIZE),
                scratch_ptr.unsafe_offset(2 * MAP_SIZE),
                scratch_ptr.unsafe_offset(3 * MAP_SIZE),
                map_ptr,
                False,
            )
            states[e, S_PLAYER_POS] = Scalar[gpu_dtype](spawn[0])
            states[e, S_PLAYER_POS + 1] = Scalar[gpu_dtype](spawn[1])
            states[e, S_PLAYER_DIR] = Scalar[gpu_dtype](DIR_UP)
            for k in range(NUM_INTRINSICS):
                states[e, S_INTRINSICS_BASE + k] = Scalar[gpu_dtype](
                    INTRINSIC_MAX
                )
            states[e, S_LIGHT_LEVEL] = Scalar[gpu_dtype](0.7969252)
            states[e, S_IS_SLEEPING] = Scalar[gpu_dtype](0.0)
            states[e, S_TIMESTEP] = Scalar[gpu_dtype](0.0)
            dones[e] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[selective_reset_wrapper](
            states,
            dones,
            scratch,
            seed_scalar,
            grid_dim=(BLOCKS,),
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
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var rewards = tile[BATCH_SIZE](rewards_buf)
        var dones = tile[BATCH_SIZE](dones_buf)
        var terminated_out = tile[BATCH_SIZE](terminated_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        var actions = tile[BATCH_SIZE](actions_buf)
        var seed_scalar = Scalar[DType.uint64](rng_seed)

        @always_inline
        def step_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[gpu_dtype, BATCH_SIZE],
            rewards: Tile[gpu_dtype, BATCH_SIZE],
            dones: Tile[gpu_dtype, BATCH_SIZE],
            terminated_out: Tile[gpu_dtype, BATCH_SIZE],
            obs: Tile[gpu_dtype, BATCH_SIZE, OBS_DIM],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return

            var action = Int(actions[e])
            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(
                e
            ) + UInt64(1)
            var rng = PhiloxRandom(seed=per_env_seed, offset=0)
            var state_ptr = states.ptr.unsafe_offset(e * STATE_SIZE)
            var result = apply_step_inline(state_ptr, action, rng)

            rewards[e] = Scalar[gpu_dtype](result[0])
            dones[e] = Scalar[gpu_dtype](1.0) if result[1] else Scalar[
                gpu_dtype
            ](0.0)
            terminated_out[e] = Scalar[gpu_dtype](0.0)

            # Symbolic obs: write into this env's slice of the obs buffer.
            var obs_ptr = obs.ptr.unsafe_offset(e * OBS_DIM)
            extract_obs_inline(state_ptr, obs_ptr)

        ctx.enqueue_function[step_wrapper](
            states,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            seed_scalar,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

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
        # `extract_obs_inline` reads state through the mutable `State` pointer
        # alias (= Pointer[..., MutAnyOrigin]), so the view must be mut.
        # Rebind the read-only buffer through a mut local to get a mut=True
        # concrete-origin view (no unsafe_ptr; widens into the MutAnyOrigin param).
        var states_buf_mut = states_buf
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf_mut)
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def extract_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            obs: Tile[gpu_dtype, BATCH_SIZE, OBS_DIM],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return
            var state_ptr = states.ptr.unsafe_offset(e * STATE_SIZE)
            var obs_ptr = obs.ptr.unsafe_offset(e * OBS_DIM)
            extract_obs_inline(state_ptr, obs_ptr)

        ctx.enqueue_function[extract_wrapper](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype],) raises:
        pass

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        pass

    # ========================================================================
    # RenderableEnv — SDL3 sprite-based, player-centered 9×7 view + HUD
    # ========================================================================
    #
    # Window layout (TILE_PX = 48, sprites 16×16 upscaled 3×):
    #   - View area: VIEW_W × TILE_PX = 432 wide, VIEW_H × TILE_PX = 336 tall
    #   - HUD strip: 144 px tall
    #   → window = 432 × 480
    #
    # Each tile draws as one src-rect from a packed sprite sheet uploaded
    # once per frame (matches the chess renderer pattern). The sprite sheet
    # is loaded once at init_renderer time from PNGs in `assets/`. If PNG
    # loading fails (no PIL), the renderer falls back to colored rects.

    comptime TILE_PX: Int = 48
    comptime VIEW_PX_W: Int = VIEW_W * Self.TILE_PX  # 432
    comptime VIEW_PX_H: Int = VIEW_H * Self.TILE_PX  # 336
    comptime HUD_PX_H: Int = 144
    comptime WIN_PX_W: Int = Self.VIEW_PX_W           # 432
    comptime WIN_PX_H: Int = Self.VIEW_PX_H + Self.HUD_PX_H  # 480

    # Asset directory — relative to project root (pixi invokes from there).
    comptime ASSET_DIR: String = "noeira/envs/craftax_classic/assets"

    @staticmethod
    @always_inline
    def _inv_sprite(slot: Int) -> Int:
        """Map an inventory slot index to a sprite index."""
        # Raw materials reuse the block sprite for visual consistency, with
        # one exception: INV_WOOD shows the log sprite (not placed wood).
        if slot == INV_WOOD:
            return SPR_INV_WOOD
        elif slot == INV_STONE:
            return 4  # SPR_STONE
        elif slot == INV_COAL:
            return 8  # SPR_COAL
        elif slot == INV_IRON:
            return 9  # SPR_IRON
        elif slot == INV_DIAMOND:
            return 10  # SPR_DIAMOND
        elif slot == INV_SAPLING:
            return SPR_INV_SAPLING
        elif slot == INV_WOOD_PICKAXE:
            return SPR_INV_WOOD_PICKAXE
        elif slot == INV_STONE_PICKAXE:
            return SPR_INV_STONE_PICKAXE
        elif slot == INV_IRON_PICKAXE:
            return SPR_INV_IRON_PICKAXE
        elif slot == INV_WOOD_SWORD:
            return SPR_INV_WOOD_SWORD
        elif slot == INV_STONE_SWORD:
            return SPR_INV_STONE_SWORD
        elif slot == INV_IRON_SWORD:
            return SPR_INV_IRON_SWORD
        else:
            return 0

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    @always_inline
    def _block_color(block_id: Int) -> SDL_Color:
        """RGB palette for the 17 Craftax-Classic block types."""
        if block_id == BLOCK_OUT_OF_BOUNDS:
            return SDL_Color(15, 15, 25, 255)
        elif block_id == BLOCK_GRASS:
            return SDL_Color(110, 175, 95, 255)
        elif block_id == BLOCK_WATER:
            return SDL_Color(50, 95, 175, 255)
        elif block_id == BLOCK_STONE:
            return SDL_Color(140, 140, 140, 255)
        elif block_id == BLOCK_TREE:
            return SDL_Color(40, 90, 35, 255)
        elif block_id == BLOCK_WOOD:
            return SDL_Color(130, 80, 35, 255)
        elif block_id == BLOCK_PATH:
            return SDL_Color(170, 150, 100, 255)
        elif block_id == BLOCK_COAL:
            return SDL_Color(45, 45, 50, 255)
        elif block_id == BLOCK_IRON:
            return SDL_Color(195, 165, 130, 255)
        elif block_id == BLOCK_DIAMOND:
            return SDL_Color(110, 220, 240, 255)
        elif block_id == BLOCK_CRAFTING_TABLE:
            return SDL_Color(180, 120, 60, 255)
        elif block_id == BLOCK_FURNACE:
            return SDL_Color(80, 70, 70, 255)
        elif block_id == BLOCK_SAND:
            return SDL_Color(220, 200, 130, 255)
        elif block_id == BLOCK_LAVA:
            return SDL_Color(230, 80, 30, 255)
        elif block_id == BLOCK_PLANT:
            return SDL_Color(170, 220, 100, 255)
        elif block_id == BLOCK_RIPE_PLANT:
            return SDL_Color(220, 220, 60, 255)
        else:
            return SDL_Color(255, 0, 255, 255)  # debug magenta

    @staticmethod
    @always_inline
    def _inv_label(slot: Int) -> String:
        if slot == INV_WOOD:
            return "W"
        elif slot == INV_STONE:
            return "S"
        elif slot == INV_COAL:
            return "C"
        elif slot == INV_IRON:
            return "I"
        elif slot == INV_DIAMOND:
            return "D"
        elif slot == INV_SAPLING:
            return "s"
        elif slot == INV_WOOD_PICKAXE:
            return "wP"
        elif slot == INV_STONE_PICKAXE:
            return "sP"
        elif slot == INV_IRON_PICKAXE:
            return "iP"
        elif slot == INV_WOOD_SWORD:
            return "wS"
        elif slot == INV_STONE_SWORD:
            return "sS"
        elif slot == INV_IRON_SWORD:
            return "iS"
        else:
            return "?"

    def init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(
            Renderer2D(
                width=Self.WIN_PX_W,
                height=Self.WIN_PX_H,
                fps=30,
                title=String("Craftax-Classic"),
            )
        )
        self._renderer_initialized = True
        # Load sprite sheet. If PIL is missing, fall back to colored rects.
        if not self._has_sprites:
            try:
                self._sprite_pixels = build_sprite_sheet(Self.ASSET_DIR)
                self._has_sprites = True
            except e:
                print("Craftax: sprite load failed (", String(e), ")")
                print("  falling back to colored-rect rendering")
                self._has_sprites = False
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._render(self._renderer.value()[])

    def _render(self, mut renderer: Renderer2D):
        """Sprite-based render: player-centered 9×7 view + HUD."""
        from noeira.render.sdl import (
            create_surface_from,
            create_texture_from_surface,
            render_texture,
            destroy_surface,
            set_texture_blend_mode,
            set_texture_scale_mode,
            destroy_texture,
            Surface,
            Texture,
            FRect,
            PixelFormat,
            BlendMode,
            ScaleMode,
        )

        var bg = SDL_Color(8, 8, 16, 255)
        if not renderer.begin_frame_with_color(bg):
            return

        comptime TS: Int = Self.TILE_PX

        var py = Int(self.state[S_PLAYER_POS])
        var px = Int(self.state[S_PLAYER_POS + 1])
        var pdir = Int(self.state[S_PLAYER_DIR])
        var sleeping = self.state[S_IS_SLEEPING] > Scalar[Self.dtype](0.5)

        # View origin in world coordinates (top-left tile).
        var ox = px - VIEW_W // 2
        var oy = py - VIEW_H // 2

        # Day/night dim: keep bright in day, darken at night. Lava/torch are
        # rendered at full brightness regardless (skipped for simplicity).
        var light = Float64(self.state[S_LIGHT_LEVEL])
        if light < 0.3:
            light = 0.3
        if light > 1.0:
            light = 1.0

        # Upload the sprite sheet as one SDL3 texture for this frame.
        var has_texture = False
        var texture: Optional[
            Pointer[Texture, MutAnyOrigin]
        ] = None
        if self._has_sprites:
            try:
                var surface = create_surface_from(
                    c_int(SPRITE_SHEET_WIDTH),
                    c_int(SPRITE_SHEET_HEIGHT),
                    PixelFormat.PIXELFORMAT_RGBA32,
                    rebind[Pointer[NoneType, MutAnyOrigin]](
                        self._sprite_pixels.value()
                    ),
                    c_int(SPRITE_SHEET_WIDTH * SPRITE_BPP),
                )
                texture = create_texture_from_surface(
                    renderer.sdl_renderer.value(), surface
                )
                set_texture_blend_mode(texture.value(), BlendMode.BLENDMODE_BLEND)
                try:
                    set_texture_scale_mode(texture.value(), ScaleMode.SCALEMODE_NEAREST)
                except:
                    pass
                destroy_surface(surface)
                has_texture = True
            except:
                pass

        # Helper: blit sprite `idx` at (dst_x, dst_y) with size `dst_size`.
        def _blit(idx: Int, dst_x: Int, dst_y: Int, dst_size: Int) {imm}:
            if not has_texture:
                return
            var src_alloc = alloc[FRect]({count = 1})
            var src = src_alloc.unsafe_ptr()
            src[] = FRect(
                c_float(idx * SPRITE_SIZE),
                c_float(0),
                c_float(SPRITE_SIZE),
                c_float(SPRITE_SIZE),
            )
            var dst_alloc = alloc[FRect]({count = 1})
            var dst = dst_alloc.unsafe_ptr()
            dst[] = FRect(
                c_float(dst_x),
                c_float(dst_y),
                c_float(dst_size),
                c_float(dst_size),
            )
            try:
                render_texture(
                    renderer.sdl_renderer.value(),
                    texture.value(),
                    rebind[Pointer[FRect, ImmutAnyOrigin]](src),
                    rebind[Pointer[FRect, ImmutAnyOrigin]](dst),
                )
            except:
                pass
            dealloc(src_alloc^)
            dealloc(dst_alloc^)

        # --- Tiles in the 9×7 view ---
        for vy in range(VIEW_H):
            for vx in range(VIEW_W):
                var wy = oy + vy
                var wx = ox + vx
                var dst_x = vx * TS
                var dst_y = vy * TS
                var block_id: Int = 1  # SPR_OOB
                var in_bounds = (
                    wy >= 0 and wy < MAP_H and wx >= 0 and wx < MAP_W
                )
                if in_bounds:
                    block_id = Int(self.state[s_map(wy, wx)])
                if has_texture:
                    _blit(block_id, dst_x, dst_y, TS)
                else:
                    var c = Self._block_color(block_id)
                    var cr = Int(Float64(Int(c.r)) * light)
                    var cg = Int(Float64(Int(c.g)) * light)
                    var cb = Int(Float64(Int(c.b)) * light)
                    renderer.draw_rect(
                        dst_x, dst_y, TS, TS,
                        SDL_Color(UInt8(cr), UInt8(cg), UInt8(cb), 255),
                    )

        # --- Plants (only if alive and inside view) ---
        for i in range(MAX_PLANTS):
            if self.state[s_plant_mask(i)] < Scalar[Self.dtype](0.5):
                continue
            var ply = Int(self.state[s_plant(i, PLANT_FY)])
            var plx = Int(self.state[s_plant(i, PLANT_FX)])
            var vy = ply - oy
            var vx = plx - ox
            if vy < 0 or vy >= VIEW_H or vx < 0 or vx >= VIEW_W:
                continue
            var age = Int(self.state[s_plant(i, PLANT_FAGE)])
            var sprite = SPR_PLANT_RIPE if age >= PLANT_RIPEN_AGE else SPR_PLANT_YOUNG
            if has_texture:
                _blit(sprite, vx * TS, vy * TS, TS)

        # --- Mobs (each: skip if dead or outside view) ---
        def _blit_mob(my: Int, mx: Int, sprite: Int) {imm}:
            var vy = my - oy
            var vx = mx - ox
            if vy < 0 or vy >= VIEW_H or vx < 0 or vx >= VIEW_W:
                return
            if has_texture:
                _blit(sprite, vx * TS, vy * TS, TS)

        for i in range(MAX_COWS):
            if self.state[s_cow(i, MOB_HP)] > Scalar[Self.dtype](0.0):
                _blit_mob(
                    Int(self.state[s_cow(i, MOB_FY)]),
                    Int(self.state[s_cow(i, MOB_FX)]),
                    SPR_COW,
                )
        for i in range(MAX_ZOMBIES):
            if self.state[s_zombie(i, MOB_HP)] > Scalar[Self.dtype](0.0):
                _blit_mob(
                    Int(self.state[s_zombie(i, MOB_FY)]),
                    Int(self.state[s_zombie(i, MOB_FX)]),
                    SPR_ZOMBIE,
                )
        for i in range(MAX_SKELETONS):
            if self.state[s_skeleton(i, MOB_HP)] > Scalar[Self.dtype](0.0):
                _blit_mob(
                    Int(self.state[s_skeleton(i, MOB_FY)]),
                    Int(self.state[s_skeleton(i, MOB_FX)]),
                    SPR_SKELETON,
                )
        for i in range(MAX_ARROWS):
            if self.state[s_arrow(i, MOB_HP)] > Scalar[Self.dtype](0.0):
                _blit_mob(
                    Int(self.state[s_arrow(i, MOB_FY)]),
                    Int(self.state[s_arrow(i, MOB_FX)]),
                    Self._arrow_sprite_for(
                        Int(self.state[s_arrow(i, ARROW_FDIR)])
                    ),
                )

        # --- Player (always at view center) ---
        var player_sprite = Self._player_sprite_for(pdir, sleeping)
        var center_vy = VIEW_H // 2
        var center_vx = VIEW_W // 2
        if has_texture:
            _blit(player_sprite, center_vx * TS, center_vy * TS, TS)

        # ====================================================================
        # HUD — 144 px tall, three rows.
        # ====================================================================
        comptime HUD_Y: Int = Self.VIEW_PX_H
        comptime HUD_H: Int = Self.HUD_PX_H
        var hud_bg = SDL_Color(28, 28, 38, 255)
        renderer.draw_rect(0, HUD_Y, Self.WIN_PX_W, HUD_H, hud_bg)
        var sep = SDL_Color(80, 80, 110, 255)
        renderer.draw_rect(0, HUD_Y, Self.WIN_PX_W, 1, sep)
        var text_color = SDL_Color(220, 220, 230, 255)
        var dim_color = SDL_Color(100, 100, 120, 255)

        # Row 1 (intrinsics): icon (24×24) + bar + value, 4 stats across.
        var row1_y = HUD_Y + 10
        var icon_sz = 24
        var bar_w = 60
        var bar_h = 8
        var slot_pitch = Self.WIN_PX_W // 4  # 108
        var bar_bg = SDL_Color(30, 30, 40, 255)
        var bar_frame = SDL_Color(80, 80, 100, 255)

        # ⚠ LEGACY CLOSURE ON PURPOSE. A unified closure would need `mut renderer`
        # while also capturing `_blit`, which holds `renderer` immutably — the
        # compiler rejects that aliasing at the closure's creation.
        @__parameter
        def _draw_intrinsic(k: Int, sprite: Int, color: SDL_Color):
            var val = Int(self.state[S_INTRINSICS_BASE + k])
            var x0 = k * slot_pitch + 6
            if has_texture:
                _blit(sprite, x0, row1_y, icon_sz)
            var bx = x0 + icon_sz + 4
            var by = row1_y + (icon_sz - bar_h) // 2
            renderer.draw_rect(bx - 1, by - 1, bar_w + 2, bar_h + 2, bar_frame)
            renderer.draw_rect(bx, by, bar_w, bar_h, bar_bg)
            var fill = (val * bar_w) // INTRINSIC_MAX
            if fill > 0:
                renderer.draw_rect(bx, by, fill, bar_h, color)
            renderer.draw_text(
                String(val) + "/" + String(INTRINSIC_MAX),
                bx, by + bar_h + 2, text_color,
            )

        _draw_intrinsic(
            INTRINSIC_HEALTH, SPR_ICON_HEALTH,
            SDL_Color(220, 60, 60, 255),
        )
        _draw_intrinsic(
            INTRINSIC_FOOD, SPR_ICON_FOOD,
            SDL_Color(220, 140, 60, 255),
        )
        _draw_intrinsic(
            INTRINSIC_DRINK, SPR_ICON_DRINK,
            SDL_Color(60, 140, 220, 255),
        )
        _draw_intrinsic(
            INTRINSIC_ENERGY, SPR_ICON_ENERGY,
            SDL_Color(220, 220, 60, 255),
        )

        # Row 2 + 3 (inventory): 6 slots × 2 rows = 12 inventory icons.
        var inv_row_y0 = HUD_Y + 50
        var inv_pitch_x = Self.WIN_PX_W // 6  # 72
        var inv_pitch_y = 46
        for k in range(NUM_INVENTORY):
            var row = k // 6
            var col = k % 6
            var x0 = col * inv_pitch_x + (inv_pitch_x - icon_sz) // 2
            var y0 = inv_row_y0 + row * inv_pitch_y
            var qty = Int(self.state[S_INV_BASE + k])
            if has_texture:
                _blit(Self._inv_sprite(k), x0, y0, icon_sz)
            var lc = text_color if qty > 0 else dim_color
            renderer.draw_text(
                String(qty),
                x0 + icon_sz + 2,
                y0 + icon_sz - 8,
                lc,
            )

        # Step / achievements / light state — bottom-left of HUD.
        var ach_count = 0
        for k in range(NUM_ACHIEVEMENTS):
            if self.state[S_ACHIEVEMENTS_BASE + k] > Scalar[Self.dtype](0.5):
                ach_count += 1
        var foot_y = HUD_Y + HUD_H - 12
        var foot_text = (
            "Step "
            + String(Int(self.state[S_TIMESTEP]))
            + "  Ach "
            + String(ach_count)
            + "/"
            + String(NUM_ACHIEVEMENTS)
            + "  Light "
            + fit(String(Float64(Int(light * 100.0)) / 100.0), 4)
        )
        if sleeping:
            foot_text = foot_text + "  Sleeping"
        renderer.draw_text(foot_text, 8, foot_y, text_color)

        # Cleanup per-frame texture.
        if has_texture:
            try:
                destroy_texture(texture.value())
            except:
                pass

        renderer.flip()

    def close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False
        if self._has_sprites:
            self._sprite_pixels.value().unsafe_free()
            self._has_sprites = False

    def is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return not self._renderer.value()[].get_should_quit()

    def check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].get_should_quit()

    def start_recording(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].start_recording(filename, fps, skip)

    def stop_recording(mut self) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].stop_recording()

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False
