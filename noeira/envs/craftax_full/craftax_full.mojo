"""Full Craftax env (CPU).

Phase 7C/35 scope:
  - `CraftaxFullEnv` struct conforming to `BoxDiscreteActionEnv` and
    `RenderableEnv` (the GPU trait + pixel obs land in #41).
  - `reset_with_seed`: zero state, gen 9-floor world, seed initial
    intrinsics / inventory / direction / monsters_killed quota.
  - `step` → `apply_step_inline` from `game_logic.mojo`.
  - Symbolic obs via `encode_symbolic_obs` (Task #36).
  - SDL3 sprite-based renderer (Task #40) — 9 floors, 24 mob species,
    projectiles, boss vulnerability, item overlays, and a HUD that mirrors
    the reference Python renderer.
"""

from std.memory import dealloc, alloc
from std.ffi import c_int, c_float
from std.random.philox import Random as PhiloxRandom

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from noeira.nn.constants import DT as gpu_dtype
from noeira.render import Renderer2D, SDL_Color

from .constants import (
    MAP_H,
    MAP_W,
    MAP_SIZE_PER_FLOOR,
    VIEW_H,
    VIEW_W,
    NUM_INTRINSICS,
    NUM_INTRINSICS_F,
    NUM_ATTRIBUTES,
    NUM_ACTIONS,
    NUM_ACHIEVEMENTS,
    NUM_INVENTORY,
    NUM_POTIONS,
    NUM_FLOORS,
    OBS_DIM,
    INTRINSIC_MAX,
    DAY_LENGTH,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
    DIR_DOWN,
    MAX_TIMESTEPS,
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    INTRINSIC_MANA,
    INTRINSIC_IS_SLEEPING,
    BOSS_FIGHT_SPAWN_TURNS,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    ATTR_XP,
    ATTR_DEXTERITY,
    ATTR_STRENGTH,
    ATTR_INTELLIGENCE,
    INV_WOOD,
    INV_STONE,
    INV_COAL,
    INV_IRON,
    INV_DIAMOND,
    INV_SAPLING,
    INV_PICKAXE,
    INV_SWORD,
    INV_BOW,
    INV_ARROWS,
    INV_ARMOUR_HEAD,
    INV_ARMOUR_BODY,
    INV_ARMOUR_LEGS,
    INV_ARMOUR_FEET,
    INV_TORCHES,
    INV_RUBY,
    INV_SAPPHIRE,
    INV_BOOKS,
    INV_POTIONS_BASE,
    BLOCK_OUT_OF_BOUNDS,
    BLOCK_NECROMANCER,
    BLOCK_NECROMANCER_VULNERABLE,
    ITEM_LADDER_DOWN,
    ITEM_LADDER_DOWN_BLOCKED,
    MAX_MELEE_MOBS,
    MAX_PASSIVE_MOBS,
    MAX_RANGED_MOBS,
    MAX_MOB_PROJECTILES,
    MAX_PLAYER_PROJECTILES,
    MOB_FY,
    MOB_FX,
    MOB_MASK,
    MOB_TYPE_ID,
    PROJ_FDIR_Y,
    PROJ_FDIR_X,
    PROJ_ARROW,
    PROJ_ARROW2,
    NUM_SPELLS,
    SPELL_FIREBALL,
    SPELL_ICEBALL,
    NUM_ARMOUR_ENCHANTS,
)
from .state import (
    STATE_SIZE,
    S_PLAYER_POS,
    S_PLAYER_LEVEL,
    S_PLAYER_DIR,
    S_LIGHT_LEVEL,
    S_TIMESTEP,
    S_BOSS_PROGRESS,
    S_BOSS_TIMESTEPS,
    S_INTRINSICS_BASE,
    S_INV_BASE,
    S_ACHIEVEMENTS_BASE,
    S_SWORD_ENCHANT,
    S_BOW_ENCHANT,
    s_intrinsic,
    s_attribute,
    s_monsters_killed,
    s_potion_mapping,
    s_inv,
    s_learned_spell,
    s_map,
    s_item_map,
    s_light_map,
    s_melee_mob,
    s_passive_mob,
    s_ranged_mob,
    s_mob_projectile,
    s_player_projectile,
)
from .world_gen import (
    generate_full_world,
    generate_full_world_inline,
    calculate_light_level,
)
from .game_logic import apply_step_inline
from .symbolic_obs import encode_symbolic_obs
from .craftax_full_sprites import (
    build_sprite_sheet,
    SPRITE_SIZE,
    SHEET_WIDTH as SPRITE_SHEET_WIDTH,
    SHEET_HEIGHT as SPRITE_SHEET_HEIGHT,
    SPRITE_BPP,
    SPR_ITEM_BASE,
    SPR_ITEM_LADDER_DOWN,
    SPR_ITEM_LADDER_DOWN_BLOCKED,
    SPR_PASSIVE_BASE,
    SPR_MELEE_BASE,
    SPR_RANGED_BASE,
    SPR_PROJ_BASE,
    SPR_ARROW_LEFT,
    SPR_ARROW_RIGHT,
    SPR_ARROW_UP,
    SPR_ARROW_DOWN,
    SPR_PLAYER_LEFT,
    SPR_PLAYER_RIGHT,
    SPR_PLAYER_UP,
    SPR_PLAYER_DOWN,
    SPR_PLAYER_SLEEP,
    SPR_PICKAXE_BASE,
    SPR_SWORD_BASE,
    SPR_BOW,
    SPR_ARMOUR_BASE,
    SPR_INV_LOG,
    SPR_INV_SAPLING,
    SPR_INV_TORCH,
    SPR_INV_BOOK,
    SPR_POTION_BASE,
    SPR_SPELL_FIREBALL,
    SPR_SPELL_ICEBALL,
    SPR_ICON_HEALTH,
    SPR_ICON_FOOD,
    SPR_ICON_DRINK,
    SPR_ICON_ENERGY,
    SPR_ICON_MANA,
    SPR_ICON_XP,
    SPR_ICON_DEX,
    SPR_ICON_STR,
    SPR_ICON_INT,
)
from noeira.core.tiles import Tile, ImmTile, tile


# ============================================================================
# Trait-required state / action types
# ============================================================================

@fieldwise_init
struct CraftaxFullState(Copyable, ImplicitlyCopyable, Movable, State):
    var index: Int

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit move: Self):
        self.index = move.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct CraftaxFullAction(Action, Copyable, ImplicitlyCopyable, Movable):
    var value: Int

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    def __init__(out self, *, deinit move: Self):
        self.value = move.value


# ============================================================================
# CraftaxFullEnv (CPU)
# ============================================================================


struct CraftaxFullEnv[DTYPE: DType = DType.float32](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Full Craftax env. CPU + GPU batched kernels (#42)."""

    comptime dtype = Self.DTYPE
    comptime StateType = CraftaxFullState
    comptime ActionType = CraftaxFullAction

    # GPUDiscreteEnv constants
    comptime STATE_SIZE: Int = STATE_SIZE
    comptime OBS_DIM: Int = OBS_DIM
    comptime NUM_ACTIONS: Int = NUM_ACTIONS
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

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

    # ------------------------------------------------------------------
    # CPU reset / step
    # ------------------------------------------------------------------

    def reset(mut self) -> Self.StateType:
        self._rng_counter += 1
        return self.reset_with_seed(self._rng_counter)

    def reset_with_seed(mut self, seed: UInt64) -> Self.StateType:
        # Zero everything.
        for i in range(STATE_SIZE):
            self.state[i] = Scalar[Self.dtype](0.0)

        # Generate 9-floor world directly into state.
        var state_ptr = self.state.unsafe_ptr().unsafe_bitcast[Float32]()
        var pos = generate_full_world(seed, state_ptr.as_unsafe_any_origin())
        var py = pos[0]
        var px = pos[1]

        # Player on the overworld, facing UP.
        self.state[S_PLAYER_LEVEL] = Scalar[Self.dtype](0)
        self.state[S_PLAYER_POS] = Scalar[Self.dtype](py)
        self.state[S_PLAYER_POS + 1] = Scalar[Self.dtype](px)
        self.state[S_PLAYER_DIR] = Scalar[Self.dtype](DIR_UP)

        # Intrinsics: health/food/drink/energy/mana at max. Sleeping/resting=0.
        self.state[s_intrinsic(INTRINSIC_HEALTH)] = Scalar[Self.dtype](
            INTRINSIC_MAX
        )
        self.state[s_intrinsic(INTRINSIC_FOOD)] = Scalar[Self.dtype](
            INTRINSIC_MAX
        )
        self.state[s_intrinsic(INTRINSIC_DRINK)] = Scalar[Self.dtype](
            INTRINSIC_MAX
        )
        self.state[s_intrinsic(INTRINSIC_ENERGY)] = Scalar[Self.dtype](
            INTRINSIC_MAX
        )
        self.state[s_intrinsic(INTRINSIC_MANA)] = Scalar[Self.dtype](
            INTRINSIC_MAX
        )

        # Attributes start at 1 (dex/str/intel), XP at 0.
        self.state[s_attribute(ATTR_DEXTERITY)] = Scalar[Self.dtype](1)
        self.state[s_attribute(ATTR_STRENGTH)] = Scalar[Self.dtype](1)
        self.state[s_attribute(ATTR_INTELLIGENCE)] = Scalar[Self.dtype](1)

        # First floor's ladder starts open.
        self.state[s_monsters_killed(0)] = Scalar[Self.dtype](
            MONSTERS_KILLED_TO_CLEAR_LEVEL + 2
        )

        # Potion mapping: identity for now. Real impl picks a permutation
        # per episode (random color → effect mapping). Identity keeps
        # the deterministic test path simple.
        for k in range(NUM_POTIONS):
            self.state[s_potion_mapping(k)] = Scalar[Self.dtype](k)

        # Boss spawn cooldown starts at the configured number of turns.
        self.state[S_BOSS_TIMESTEPS] = Scalar[Self.dtype](
            BOSS_FIGHT_SPAWN_TURNS
        )

        self.state[S_LIGHT_LEVEL] = Scalar[Self.dtype](
            calculate_light_level(0)
        )
        self.state[S_TIMESTEP] = Scalar[Self.dtype](0.0)

        self.done = False
        return CraftaxFullState(index=0)

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            CraftaxFullState(index=Int(self.state[S_TIMESTEP])),
            result[0],
            result[1],
        )

    def _step_impl(
        mut self, action: Int
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        self._rng_counter += 1
        var rng = PhiloxRandom(seed=self._rng_counter, offset=0)
        var state_ptr = self.state.unsafe_ptr().unsafe_bitcast[Float32]()
        var result = apply_step_inline(
            state_ptr.as_unsafe_any_origin(), action, rng
        )
        self.done = result[1]
        return (Scalar[Self.dtype](result[0]), self.done)

    # ------------------------------------------------------------------
    # Env trait methods
    # ------------------------------------------------------------------

    def get_state(mut self) -> Self.StateType:
        return CraftaxFullState(index=Int(self.state[S_TIMESTEP]))

    def close(mut self):
        pass

    def action_from_index(self, action_idx: Int) -> Self.ActionType:
        return CraftaxFullAction(value=action_idx)

    def num_actions(self) -> Int:
        return NUM_ACTIONS

    def obs_dim(self) -> Int:
        return OBS_DIM

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: Self.StateType) -> Int:
        return state.index

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Build the 8268-D Craftax-Full symbolic observation."""
        var obs_arr = Array[Float32, OBS_DIM](fill=Float32(0.0))
        var obs_ptr = rebind[Pointer[Float32, MutAnyOrigin]](
            obs_arr.unsafe_ptr().unsafe_bitcast[Float32]()
        )
        var state_ptr = rebind[Pointer[Float32, MutAnyOrigin]](
            self.state.unsafe_ptr().unsafe_bitcast[Float32]()
        )
        encode_symbolic_obs(state_ptr, obs_ptr)
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
    # GPU kernels (#42) — batched reset / step / extract_obs on DeviceBuffer
    # ========================================================================
    #
    # Per-env world-gen scratch is `4 × MAP_SIZE_PER_FLOOR` floats (water,
    # mountain, path, tree noise fields). Mob arrays / inventory / RNG state
    # all live inside the per-env state slice, so the only extra workspace
    # we need is the noise scratch.

    comptime TPB: Int = 64           # threads/block — modest because each
                                     # thread does a lot of work per step
                                     # (8268-D obs encode alone is hot).
    comptime WORLD_GEN_WS_PER_ENV: Int = 4 * MAP_SIZE_PER_FLOOR  # 9216 floats

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Generate a fresh world for every env in the batch."""
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)

        var scratch_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * Self.WORLD_GEN_WS_PER_ENV
        )
        var scratch = tile[BATCH_SIZE * Self.WORLD_GEN_WS_PER_ENV](scratch_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed_scalar = Scalar[DType.uint64](rng_seed)

        @always_inline
        def reset_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            scratch: Tile[gpu_dtype, BATCH_SIZE * Self.WORLD_GEN_WS_PER_ENV],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return

            # Zero this env's state slice.
            for s in range(STATE_SIZE):
                states[e, s] = Scalar[gpu_dtype](0.0)

            # Slice per-env scratch.
            var ws_base = e * Self.WORLD_GEN_WS_PER_ENV
            var scratch_ptr = scratch.ptr.unsafe_offset(ws_base)
            var water_ptr = scratch_ptr
            var mountain_ptr = scratch_ptr.unsafe_offset(MAP_SIZE_PER_FLOOR)
            var path_ptr = scratch_ptr.unsafe_offset(2 * MAP_SIZE_PER_FLOOR)
            var tree_ptr = scratch_ptr.unsafe_offset(3 * MAP_SIZE_PER_FLOOR)
            var state_ptr = states.ptr.unsafe_offset(e * STATE_SIZE)

            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(e) + UInt64(1)
            var spawn = generate_full_world_inline(
                per_env_seed,
                state_ptr,
                water_ptr,
                mountain_ptr,
                path_ptr,
                tree_ptr,
            )

            # Standard reset payload (mirror CPU `reset_with_seed`).
            states[e, S_PLAYER_LEVEL] = Scalar[gpu_dtype](0)
            states[e, S_PLAYER_POS] = Scalar[gpu_dtype](spawn[0])
            states[e, S_PLAYER_POS + 1] = Scalar[gpu_dtype](spawn[1])
            states[e, S_PLAYER_DIR] = Scalar[gpu_dtype](DIR_UP)
            states[e, s_intrinsic(INTRINSIC_HEALTH)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_intrinsic(INTRINSIC_FOOD)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_intrinsic(INTRINSIC_DRINK)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_intrinsic(INTRINSIC_ENERGY)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_intrinsic(INTRINSIC_MANA)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_attribute(ATTR_DEXTERITY)] = Scalar[gpu_dtype](1)
            states[e, s_attribute(ATTR_STRENGTH)] = Scalar[gpu_dtype](1)
            states[e, s_attribute(ATTR_INTELLIGENCE)] = Scalar[gpu_dtype](1)
            states[e, s_monsters_killed(0)] = Scalar[gpu_dtype](
                MONSTERS_KILLED_TO_CLEAR_LEVEL + 2
            )
            for k in range(NUM_POTIONS):
                states[e, s_potion_mapping(k)] = Scalar[gpu_dtype](k)
            states[e, S_BOSS_TIMESTEPS] = Scalar[gpu_dtype](BOSS_FIGHT_SPAWN_TURNS)
            states[e, S_LIGHT_LEVEL] = Scalar[gpu_dtype](calculate_light_level(0))
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
        """Reset only envs where done == 1."""
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var dones = tile[BATCH_SIZE](dones_buf)

        var scratch_buf = ctx.enqueue_create_buffer[gpu_dtype](
            BATCH_SIZE * Self.WORLD_GEN_WS_PER_ENV
        )
        var scratch = tile[BATCH_SIZE * Self.WORLD_GEN_WS_PER_ENV](scratch_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed_scalar = Scalar[DType.uint64](rng_seed)

        @always_inline
        def selective_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            dones: Tile[gpu_dtype, BATCH_SIZE],
            scratch: Tile[gpu_dtype, BATCH_SIZE * Self.WORLD_GEN_WS_PER_ENV],
            seed: Scalar[DType.uint64],
        ):
            var e = Int(block_dim.x * block_idx.x + thread_idx.x)
            if e >= BATCH_SIZE:
                return
            if dones[e] <= Scalar[gpu_dtype](0.5):
                return

            for s in range(STATE_SIZE):
                states[e, s] = Scalar[gpu_dtype](0.0)

            var ws_base = e * Self.WORLD_GEN_WS_PER_ENV
            var scratch_ptr = scratch.ptr.unsafe_offset(ws_base)
            var state_ptr = states.ptr.unsafe_offset(e * STATE_SIZE)
            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(e) + UInt64(1)
            var spawn = generate_full_world_inline(
                per_env_seed,
                state_ptr,
                scratch_ptr,
                scratch_ptr.unsafe_offset(MAP_SIZE_PER_FLOOR),
                scratch_ptr.unsafe_offset(2 * MAP_SIZE_PER_FLOOR),
                scratch_ptr.unsafe_offset(3 * MAP_SIZE_PER_FLOOR),
            )
            states[e, S_PLAYER_LEVEL] = Scalar[gpu_dtype](0)
            states[e, S_PLAYER_POS] = Scalar[gpu_dtype](spawn[0])
            states[e, S_PLAYER_POS + 1] = Scalar[gpu_dtype](spawn[1])
            states[e, S_PLAYER_DIR] = Scalar[gpu_dtype](DIR_UP)
            states[e, s_intrinsic(INTRINSIC_HEALTH)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_intrinsic(INTRINSIC_FOOD)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_intrinsic(INTRINSIC_DRINK)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_intrinsic(INTRINSIC_ENERGY)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_intrinsic(INTRINSIC_MANA)] = Scalar[gpu_dtype](INTRINSIC_MAX)
            states[e, s_attribute(ATTR_DEXTERITY)] = Scalar[gpu_dtype](1)
            states[e, s_attribute(ATTR_STRENGTH)] = Scalar[gpu_dtype](1)
            states[e, s_attribute(ATTR_INTELLIGENCE)] = Scalar[gpu_dtype](1)
            states[e, s_monsters_killed(0)] = Scalar[gpu_dtype](
                MONSTERS_KILLED_TO_CLEAR_LEVEL + 2
            )
            for k in range(NUM_POTIONS):
                states[e, s_potion_mapping(k)] = Scalar[gpu_dtype](k)
            states[e, S_BOSS_TIMESTEPS] = Scalar[gpu_dtype](BOSS_FIGHT_SPAWN_TURNS)
            states[e, S_LIGHT_LEVEL] = Scalar[gpu_dtype](calculate_light_level(0))
            states[e, S_TIMESTEP] = Scalar[gpu_dtype](0.0)
            dones[e] = Scalar[gpu_dtype](0.0)

        ctx.enqueue_function[selective_wrapper](
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
        """Apply one step + symbolic obs encode for every env."""
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
            var per_env_seed = UInt64(seed) * UInt64(BATCH_SIZE) + UInt64(e) + UInt64(1)
            var rng = PhiloxRandom(seed=per_env_seed, offset=0)
            var state_ptr = states.ptr.unsafe_offset(e * STATE_SIZE)
            var result = apply_step_inline(state_ptr, action, rng)

            rewards[e] = Scalar[gpu_dtype](result[0])
            dones[e] = (
                Scalar[gpu_dtype](1.0) if result[1] else Scalar[gpu_dtype](0.0)
            )
            terminated_out[e] = Scalar[gpu_dtype](0.0)

            # Symbolic obs (8268-D). Reuses the same per-env CPU helper —
            # all pointer arithmetic, no Python.
            var obs_ptr = obs.ptr.unsafe_offset(e * OBS_DIM)
            encode_symbolic_obs(state_ptr, obs_ptr)

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
        """Encode the symbolic obs into `obs_buf` for every env."""
        # `encode_symbolic_obs` reads state through the mutable `State` pointer
        # alias, so the view must be mut. Rebind the read-only buffer through a
        # mut local to get a mut=True concrete-origin view (no unsafe_ptr).
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
            encode_symbolic_obs(state_ptr, obs_ptr)

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
    # RenderableEnv — SDL3 sprite-based, player-centered 9×11 view + HUD
    # ========================================================================
    #
    # Window layout (TILE_PX = 48, sprites 16×16 upscaled 3×):
    #   - View area: VIEW_W × TILE_PX = 528 wide, VIEW_H × TILE_PX = 432 tall
    #   - HUD strip: 192 px tall (5 intrinsic bars + inventory grid + footer)
    #   → window = 528 × 624
    #
    # Same sprite-sheet upload pattern as Craftax-Classic. The sheet is
    # loaded once at init_renderer time from PNGs in `assets/`. If PNG
    # loading fails (no PIL), the renderer falls back to colored rects so
    # the call sites still work.

    comptime TILE_PX: Int = 48
    comptime VIEW_PX_W: Int = VIEW_W * Self.TILE_PX       # 528
    comptime VIEW_PX_H: Int = VIEW_H * Self.TILE_PX       # 432
    comptime HUD_PX_H: Int = 192
    comptime WIN_PX_W: Int = Self.VIEW_PX_W                # 528
    comptime WIN_PX_H: Int = Self.VIEW_PX_H + Self.HUD_PX_H  # 624

    comptime ASSET_DIR: String = "noeira/envs/craftax_full/assets"

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
    def _projectile_sprite_for(species: Int, dy: Int, dx: Int) -> Int:
        """Map projectile species + direction → sprite index.

        For arrow species (0 and 4) we have proper directional textures.
        Other projectiles just blit the base sprite (matches reference, which
        only rotates/flips the arrow visuals)."""
        if species == PROJ_ARROW or species == PROJ_ARROW2:
            if dx < 0:
                return SPR_ARROW_LEFT
            if dx > 0:
                return SPR_ARROW_RIGHT
            if dy < 0:
                return SPR_ARROW_UP
            return SPR_ARROW_DOWN
        # Clamp to projectile slot range; species is already 0..7.
        return SPR_PROJ_BASE + species

    def init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(
            Renderer2D(
                width=Self.WIN_PX_W,
                height=Self.WIN_PX_H,
                fps=30,
                title=String("Craftax-Full"),
            )
        )
        self._renderer_initialized = True
        if not self._has_sprites:
            try:
                self._sprite_pixels = build_sprite_sheet(Self.ASSET_DIR)
                self._has_sprites = True
            except e:
                print("Craftax-Full: sprite load failed (", String(e), ")")
                print("  falling back to colored-rect rendering")
                self._has_sprites = False
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._render(self._renderer.value()[])

    def _render(self, mut renderer: Renderer2D):
        """Sprite-based render: player-centered 9×11 view of the current
        floor + intrinsic bars + inventory grid HUD."""
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

        var floor = Int(self.state[S_PLAYER_LEVEL])
        if floor < 0:
            floor = 0
        if floor >= NUM_FLOORS:
            floor = NUM_FLOORS - 1

        var py = Int(self.state[S_PLAYER_POS])
        var px = Int(self.state[S_PLAYER_POS + 1])
        var pdir = Int(self.state[S_PLAYER_DIR])
        var sleeping = (
            self.state[s_intrinsic(INTRINSIC_IS_SLEEPING)]
            > Scalar[Self.dtype](0.5)
        )

        # View origin in world coords (top-left tile of the visible window).
        var ox = px - VIEW_W // 2
        var oy = py - VIEW_H // 2

        # Light dimming: overworld uses the day/night light_level; underground
        # floors use the per-tile light_map so torches keep visibility local.
        var global_light = Float64(self.state[S_LIGHT_LEVEL])
        if global_light < 0.3:
            global_light = 0.3
        if global_light > 1.0:
            global_light = 1.0

        var is_boss_vulnerable = (
            Int(self.state[S_BOSS_PROGRESS]) >= 3
            and Int(self.state[S_BOSS_TIMESTEPS]) == 0
        )
        var ladder_open = (
            Int(self.state[s_monsters_killed(floor)])
            >= MONSTERS_KILLED_TO_CLEAR_LEVEL
        )

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
                    set_texture_scale_mode(
                        texture.value(), ScaleMode.SCALEMODE_NEAREST
                    )
                except:
                    pass
                destroy_surface(surface)
                has_texture = True
            except:
                pass

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

        # --- Tiles in the 9×11 view (current floor) ---
        for vy in range(VIEW_H):
            for vx in range(VIEW_W):
                var wy = oy + vy
                var wx = ox + vx
                var dst_x = vx * TS
                var dst_y = vy * TS
                var block_id: Int = BLOCK_OUT_OF_BOUNDS
                var item_id: Int = 0
                var lit = True
                var in_bounds = (
                    wy >= 0 and wy < MAP_H and wx >= 0 and wx < MAP_W
                )
                if in_bounds:
                    block_id = Int(self.state[s_map(floor, wy, wx)])
                    item_id = Int(self.state[s_item_map(floor, wy, wx)])
                    if floor != 0:
                        # Underground: respect per-tile light map.
                        var ltile = Float32(self.state[s_light_map(floor, wy, wx)])
                        lit = ltile > Float32(0.05)
                # Necromancer visibility flips by vulnerability state.
                if (
                    block_id == BLOCK_NECROMANCER
                    and is_boss_vulnerable
                ):
                    block_id = BLOCK_NECROMANCER_VULNERABLE
                # Ladders block visually until the floor is cleared.
                if item_id == ITEM_LADDER_DOWN and not ladder_open:
                    item_id = ITEM_LADDER_DOWN_BLOCKED

                if has_texture and lit:
                    _blit(block_id, dst_x, dst_y, TS)
                    if item_id != 0:
                        _blit(SPR_ITEM_BASE + item_id, dst_x, dst_y, TS)
                elif lit:
                    # Fallback colored tile (no sprites loaded).
                    renderer.draw_rect(
                        dst_x, dst_y, TS, TS,
                        SDL_Color(60, 60, 60, 255),
                    )
                else:
                    # Dark tile — opaque black.
                    renderer.draw_rect(
                        dst_x, dst_y, TS, TS,
                        SDL_Color(0, 0, 0, 255),
                    )

        # --- Mobs (per-floor, per-class). Each mob slot: mask=1 → render. ---
        def _blit_mob_at(my: Int, mx: Int, sprite: Int) {imm}:
            var vy = my - oy
            var vx = mx - ox
            if vy < 0 or vy >= VIEW_H or vx < 0 or vx >= VIEW_W:
                return
            if has_texture:
                _blit(sprite, vx * TS, vy * TS, TS)

        for i in range(MAX_MELEE_MOBS):
            if self.state[s_melee_mob(floor, i, MOB_MASK)] > Scalar[Self.dtype](0.5):
                var sp = Int(self.state[s_melee_mob(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                _blit_mob_at(
                    Int(self.state[s_melee_mob(floor, i, MOB_FY)]),
                    Int(self.state[s_melee_mob(floor, i, MOB_FX)]),
                    SPR_MELEE_BASE + sp,
                )
        for i in range(MAX_PASSIVE_MOBS):
            if self.state[s_passive_mob(floor, i, MOB_MASK)] > Scalar[Self.dtype](0.5):
                var sp = Int(self.state[s_passive_mob(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                _blit_mob_at(
                    Int(self.state[s_passive_mob(floor, i, MOB_FY)]),
                    Int(self.state[s_passive_mob(floor, i, MOB_FX)]),
                    SPR_PASSIVE_BASE + sp,
                )
        for i in range(MAX_RANGED_MOBS):
            if self.state[s_ranged_mob(floor, i, MOB_MASK)] > Scalar[Self.dtype](0.5):
                var sp = Int(self.state[s_ranged_mob(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                _blit_mob_at(
                    Int(self.state[s_ranged_mob(floor, i, MOB_FY)]),
                    Int(self.state[s_ranged_mob(floor, i, MOB_FX)]),
                    SPR_RANGED_BASE + sp,
                )
        for i in range(MAX_MOB_PROJECTILES):
            if self.state[s_mob_projectile(floor, i, MOB_MASK)] > Scalar[Self.dtype](0.5):
                var sp = Int(self.state[s_mob_projectile(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                var dy = Int(self.state[s_mob_projectile(floor, i, PROJ_FDIR_Y)])
                var dx = Int(self.state[s_mob_projectile(floor, i, PROJ_FDIR_X)])
                _blit_mob_at(
                    Int(self.state[s_mob_projectile(floor, i, MOB_FY)]),
                    Int(self.state[s_mob_projectile(floor, i, MOB_FX)]),
                    Self._projectile_sprite_for(sp, dy, dx),
                )
        for i in range(MAX_PLAYER_PROJECTILES):
            if self.state[s_player_projectile(floor, i, MOB_MASK)] > Scalar[Self.dtype](0.5):
                var sp = Int(self.state[s_player_projectile(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                var dy = Int(self.state[s_player_projectile(floor, i, PROJ_FDIR_Y)])
                var dx = Int(self.state[s_player_projectile(floor, i, PROJ_FDIR_X)])
                _blit_mob_at(
                    Int(self.state[s_player_projectile(floor, i, MOB_FY)]),
                    Int(self.state[s_player_projectile(floor, i, MOB_FX)]),
                    Self._projectile_sprite_for(sp, dy, dx),
                )

        # --- Player (always at view center) ---
        var player_sprite = Self._player_sprite_for(pdir, sleeping)
        var center_vy = VIEW_H // 2
        var center_vx = VIEW_W // 2
        if has_texture:
            _blit(player_sprite, center_vx * TS, center_vy * TS, TS)

        # ====================================================================
        # HUD — 192 px tall, four rows.
        # ====================================================================
        comptime HUD_Y: Int = Self.VIEW_PX_H
        comptime HUD_H: Int = Self.HUD_PX_H
        var hud_bg = SDL_Color(28, 28, 38, 255)
        renderer.draw_rect(0, HUD_Y, Self.WIN_PX_W, HUD_H, hud_bg)
        var sep = SDL_Color(80, 80, 110, 255)
        renderer.draw_rect(0, HUD_Y, Self.WIN_PX_W, 1, sep)
        var text_color = SDL_Color(220, 220, 230, 255)
        var dim_color = SDL_Color(100, 100, 120, 255)

        # Row 1 (intrinsics): 5 stat icons + bars + numeric value.
        # 5 hardcoded stats: HEALTH, FOOD, DRINK, ENERGY, MANA.
        var row1_y = HUD_Y + 6
        var icon_sz = 22
        var bar_w = 50
        var bar_h = 7
        var slot_pitch = Self.WIN_PX_W // 5  # 105
        var bar_bg = SDL_Color(30, 30, 40, 255)
        var bar_frame = SDL_Color(80, 80, 100, 255)

        var intrinsic_keys = [
            INTRINSIC_HEALTH, INTRINSIC_FOOD, INTRINSIC_DRINK,
            INTRINSIC_ENERGY, INTRINSIC_MANA,
        ]
        var intrinsic_sprites = [
            SPR_ICON_HEALTH, SPR_ICON_FOOD, SPR_ICON_DRINK,
            SPR_ICON_ENERGY, SPR_ICON_MANA,
        ]
        var intrinsic_colors = [
            SDL_Color(220, 60, 60, 255),
            SDL_Color(220, 140, 60, 255),
            SDL_Color(60, 140, 220, 255),
            SDL_Color(220, 220, 60, 255),
            SDL_Color(160, 80, 220, 255),
        ]
        for k in range(5):
            var val = Int(self.state[s_intrinsic(intrinsic_keys[k])])
            var x0 = k * slot_pitch + 6
            if has_texture:
                _blit(intrinsic_sprites[k], x0, row1_y, icon_sz)
            var bx = x0 + icon_sz + 4
            var by = row1_y + (icon_sz - bar_h) // 2
            renderer.draw_rect(bx - 1, by - 1, bar_w + 2, bar_h + 2, bar_frame)
            renderer.draw_rect(bx, by, bar_w, bar_h, bar_bg)
            var fill = (val * bar_w) // INTRINSIC_MAX
            if fill > 0:
                renderer.draw_rect(bx, by, fill, bar_h, intrinsic_colors[k])
            renderer.draw_text(
                String(val) + "/" + String(INTRINSIC_MAX),
                bx, by + bar_h + 2, text_color,
            )

        # Row 2: attributes (xp / dex / str / int) — small icon + count.
        var row2_y = row1_y + icon_sz + 12
        var attr_keys = [ATTR_XP, ATTR_DEXTERITY, ATTR_STRENGTH, ATTR_INTELLIGENCE]
        var attr_sprites = [SPR_ICON_XP, SPR_ICON_DEX, SPR_ICON_STR, SPR_ICON_INT]
        for slot in range(4):
            var val = Int(self.state[s_attribute(attr_keys[slot])])
            var x0 = slot * slot_pitch + 6
            if has_texture:
                _blit(attr_sprites[slot], x0, row2_y, icon_sz)
            var lc = text_color if val > 0 else dim_color
            renderer.draw_text(
                String(val), x0 + icon_sz + 4, row2_y + 4, lc,
            )
        # Floor index as a small label in slot 4.
        renderer.draw_text(
            "Floor " + String(floor),
            4 * slot_pitch + 6, row2_y + 4, text_color,
        )

        # Row 3 (inventory tools + armour): pickaxe / sword / bow / arrows /
        # torches / books + 4 armour pieces.
        var row3_y = row2_y + icon_sz + 12
        var pick_tier = Int(self.state[s_inv(INV_PICKAXE)])
        var sword_tier = Int(self.state[s_inv(INV_SWORD)])
        if pick_tier < 0:
            pick_tier = 0
        if pick_tier > 4:
            pick_tier = 4
        if sword_tier < 0:
            sword_tier = 0
        if sword_tier > 4:
            sword_tier = 4
        var row3_sprites = [
            SPR_PICKAXE_BASE + pick_tier,
            SPR_SWORD_BASE + sword_tier,
            SPR_BOW,
            SPR_ARROW_UP,
            SPR_INV_TORCH,
            SPR_INV_BOOK,
        ]
        var row3_qtys = [
            pick_tier,
            sword_tier,
            Int(self.state[s_inv(INV_BOW)]),
            Int(self.state[s_inv(INV_ARROWS)]),
            Int(self.state[s_inv(INV_TORCHES)]),
            Int(self.state[s_inv(INV_BOOKS)]),
        ]
        for slot in range(6):
            var x0 = slot * (icon_sz + 28) + 6
            if has_texture:
                _blit(row3_sprites[slot], x0, row3_y, icon_sz)
            var lc = text_color if row3_qtys[slot] > 0 else dim_color
            renderer.draw_text(
                String(row3_qtys[slot]), x0 + icon_sz + 2, row3_y + 4, lc,
            )

        # Armour pieces in slots 6..9. tier 0 = empty, 1=iron, 2=diamond.
        var armour_slots = [
            INV_ARMOUR_HEAD, INV_ARMOUR_BODY,
            INV_ARMOUR_LEGS, INV_ARMOUR_FEET,
        ]
        for piece in range(4):
            var tier = Int(self.state[s_inv(armour_slots[piece])])
            if tier < 0:
                tier = 0
            if tier > 2:
                tier = 2
            var x0 = (6 + piece) * (icon_sz + 28) + 6
            if has_texture and tier > 0:
                _blit(
                    SPR_ARMOUR_BASE + piece * 3 + tier,
                    x0, row3_y, icon_sz,
                )
            var lc = text_color if tier > 0 else dim_color
            renderer.draw_text(
                String(tier), x0 + icon_sz + 2, row3_y + 4, lc,
            )

        # Row 4: materials qty + spell unlocks.
        var row4_y = row3_y + icon_sz + 12
        var mat_sprites = [SPR_INV_LOG, 4, 8, 9, 10, 22, 21]  # log/stone/coal/iron/dmnd/ruby/sapphire
        var mat_qtys = [
            Int(self.state[s_inv(INV_WOOD)]),
            Int(self.state[s_inv(INV_STONE)]),
            Int(self.state[s_inv(INV_COAL)]),
            Int(self.state[s_inv(INV_IRON)]),
            Int(self.state[s_inv(INV_DIAMOND)]),
            Int(self.state[s_inv(INV_RUBY)]),
            Int(self.state[s_inv(INV_SAPPHIRE)]),
        ]
        for slot in range(7):
            var x0 = slot * (icon_sz + 28) + 6
            if has_texture:
                _blit(mat_sprites[slot], x0, row4_y, icon_sz)
            var lc = text_color if mat_qtys[slot] > 0 else dim_color
            renderer.draw_text(
                String(mat_qtys[slot]), x0 + icon_sz + 2, row4_y + 4, lc,
            )

        # Spell slots: show learned-spell icons (skipped if not learned).
        if has_texture:
            if self.state[s_learned_spell(SPELL_FIREBALL)] > Scalar[Self.dtype](0.5):
                _blit(SPR_SPELL_FIREBALL, 7 * (icon_sz + 28) + 6, row4_y,
                      icon_sz)
            if self.state[s_learned_spell(SPELL_ICEBALL)] > Scalar[Self.dtype](0.5):
                _blit(SPR_SPELL_ICEBALL, 8 * (icon_sz + 28) + 6, row4_y,
                      icon_sz)

        # Step / floor / achievement count / boss-vulnerable state — footer line.
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
            + fit(String(Float64(Int(global_light * 100.0)) / 100.0), 4)
        )
        if sleeping:
            foot_text = foot_text + "  Sleeping"
        if is_boss_vulnerable:
            foot_text = foot_text + "  BOSS VULNERABLE"
        renderer.draw_text(foot_text, 8, foot_y, text_color)

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

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

from noeira.core.fmt import fit