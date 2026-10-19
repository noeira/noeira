"""Craftax-Full — RGB sprite pixel observation (matches paper's spec).

Each output pixel maps to one tile in the player-centered 9×11 view (top) or
to one cell in the 4×11 inventory grid (bottom). Blocks/items/mobs/player are
rendered from real PNG sprites pre-resized to `BLOCK_PIXEL_SIZE` (10) via
nearest-neighbor — same agent-obs pipeline the original Craftax paper uses.

Layout:
    Obs canvas    : (VIEW_H + INVENTORY_OBS_HEIGHT) × VIEW_W × BPS²
                  = (9 + 4) × 11 × 100 pixels = 130 × 110 × 3 RGB
    OBS_DIM       : 3 * 130 * 110 = 42 900 (channel-first, row-major)

State: identical to `CraftaxFullEnv` (STATE_SIZE = 84 035).

GPU support is deferred (#41 lands as CPU-only first); subsequent passes
can lift the per-pixel helpers into a GPU kernel like Classic does.
"""

from std.memory import alloc, dealloc
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

from .game_logic import apply_step_inline

from .craftax_full import (
    CraftaxFullEnv,
    CraftaxFullState,
    CraftaxFullAction,
)
from .constants import (
    MAP_H,
    MAP_W,
    NUM_ACTIONS,
    VIEW_H,
    VIEW_W,
    NUM_FLOORS,
    NUM_INVENTORY,
    NUM_POTIONS,
    NUM_ACHIEVEMENTS,
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
    INTRINSIC_HEALTH,
    INTRINSIC_FOOD,
    INTRINSIC_DRINK,
    INTRINSIC_ENERGY,
    INTRINSIC_MANA,
    INTRINSIC_IS_SLEEPING,
    BLOCK_OUT_OF_BOUNDS,
    BLOCK_NECROMANCER,
    BLOCK_NECROMANCER_VULNERABLE,
    ITEM_LADDER_DOWN,
    ITEM_LADDER_DOWN_BLOCKED,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_UP,
    DIR_DOWN,
    INV_WOOD,
    INV_STONE,
    INV_COAL,
    INV_IRON,
    INV_DIAMOND,
    INV_SAPPHIRE,
    INV_RUBY,
    INV_SAPLING,
    INV_PICKAXE,
    INV_SWORD,
    INV_BOW,
    INV_ARROWS,
    INV_TORCHES,
    INV_BOOKS,
    INV_ARMOUR_HEAD,
    INV_ARMOUR_BODY,
    INV_ARMOUR_LEGS,
    INV_ARMOUR_FEET,
    INV_POTIONS_BASE,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    ATTR_XP,
    ATTR_DEXTERITY,
    ATTR_STRENGTH,
    ATTR_INTELLIGENCE,
    NUM_SPELLS,
    SPELL_FIREBALL,
    SPELL_ICEBALL,
)
from .state import (
    STATE_SIZE,
    S_PLAYER_POS,
    S_PLAYER_LEVEL,
    S_PLAYER_DIR,
    S_LIGHT_LEVEL,
    S_BOSS_PROGRESS,
    S_BOSS_TIMESTEPS,
    s_intrinsic,
    s_attribute,
    s_inv,
    s_map,
    s_item_map,
    s_light_map,
    s_monsters_killed,
    s_melee_mob,
    s_passive_mob,
    s_ranged_mob,
    s_mob_projectile,
    s_player_projectile,
    s_learned_spell,
)
from .craftax_full_sprites import (
    build_agent_atlas,
    agent_atlas_size,
    NUM_SPRITES,
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


# ============================================================================
# Pixel obs geometry
# ============================================================================

comptime BLOCK_PIXEL_SIZE: Int = 10                # Craftax BLOCK_PIXEL_SIZE_AGENT
comptime INVENTORY_OBS_HEIGHT: Int = 4             # rows of inventory cells
comptime OBS_PIX_H: Int = (VIEW_H + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE  # 130
comptime OBS_PIX_W: Int = VIEW_W * BLOCK_PIXEL_SIZE                          # 110
comptime OBS_CHANNELS: Int = 3
comptime PIXEL_OBS_DIM: Int = OBS_CHANNELS * OBS_PIX_H * OBS_PIX_W            # 42 900

comptime VIEW_PIX_H: Int = VIEW_H * BLOCK_PIXEL_SIZE                          # 90
comptime ATLAS_FLOATS: Int = NUM_SPRITES * BLOCK_PIXEL_SIZE * BLOCK_PIXEL_SIZE * 4

# Asset directory — relative to project root.
comptime ASSET_DIR: String = "noeira/envs/craftax_full/assets"


# ============================================================================
# Atlas sample / alpha composite primitives
# ============================================================================

@always_inline
def _atlas_sample(
    atlas: Pointer[Float32, MutAnyOrigin],
    sprite_idx: Int,
    ly: Int,
    lx: Int,
) -> Tuple[Float32, Float32, Float32, Float32]:
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
def _projectile_sprite_for(species: Int, dy: Int, dx: Int) -> Int:
    if species == PROJ_ARROW or species == PROJ_ARROW2:
        if dx < 0:
            return SPR_ARROW_LEFT
        if dx > 0:
            return SPR_ARROW_RIGHT
        if dy < 0:
            return SPR_ARROW_UP
        return SPR_ARROW_DOWN
    return SPR_PROJ_BASE + species


# ============================================================================
# Per-tile overlay rendering — kept in small helpers so the per-pixel kernel
# stays simple. We scan each (vy, vx) tile in `_render_view_tile` rather than
# iterating mobs in the inner pixel loop (much less work for a 9×11 view).
# ============================================================================


@always_inline
def _is_boss_vulnerable(s: Pointer[Float32, MutAnyOrigin]) -> Bool:
    return Int(s[unsafe_offset=S_BOSS_PROGRESS]) >= 3 and Int(s[unsafe_offset=S_BOSS_TIMESTEPS]) == 0


@always_inline
def _ladder_open(
    s: Pointer[Float32, MutAnyOrigin], floor: Int
) -> Bool:
    return (
        Int(s[unsafe_offset=s_monsters_killed(floor)])
        >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    )


def _mob_at(
    s: Pointer[Float32, MutAnyOrigin],
    floor: Int,
    wy: Int,
    wx: Int,
) -> Int:
    """Return sprite-idx of any mob/projectile at (floor, wy, wx), or -1.

    Render order matches the reference: passive → ranged → melee → mob proj
    → player proj. (The reference composites in the same order so later
    drawings win on overlap.)"""
    for i in range(MAX_PASSIVE_MOBS):
        if s[unsafe_offset=s_passive_mob(floor, i, MOB_MASK)] > Float32(0.5):
            if (
                Int(s[unsafe_offset=s_passive_mob(floor, i, MOB_FY)]) == wy
                and Int(s[unsafe_offset=s_passive_mob(floor, i, MOB_FX)]) == wx
            ):
                var sp = Int(s[unsafe_offset=s_passive_mob(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                return SPR_PASSIVE_BASE + sp
    for i in range(MAX_RANGED_MOBS):
        if s[unsafe_offset=s_ranged_mob(floor, i, MOB_MASK)] > Float32(0.5):
            if (
                Int(s[unsafe_offset=s_ranged_mob(floor, i, MOB_FY)]) == wy
                and Int(s[unsafe_offset=s_ranged_mob(floor, i, MOB_FX)]) == wx
            ):
                var sp = Int(s[unsafe_offset=s_ranged_mob(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                return SPR_RANGED_BASE + sp
    for i in range(MAX_MELEE_MOBS):
        if s[unsafe_offset=s_melee_mob(floor, i, MOB_MASK)] > Float32(0.5):
            if (
                Int(s[unsafe_offset=s_melee_mob(floor, i, MOB_FY)]) == wy
                and Int(s[unsafe_offset=s_melee_mob(floor, i, MOB_FX)]) == wx
            ):
                var sp = Int(s[unsafe_offset=s_melee_mob(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                return SPR_MELEE_BASE + sp
    for i in range(MAX_MOB_PROJECTILES):
        if s[unsafe_offset=s_mob_projectile(floor, i, MOB_MASK)] > Float32(0.5):
            if (
                Int(s[unsafe_offset=s_mob_projectile(floor, i, MOB_FY)]) == wy
                and Int(s[unsafe_offset=s_mob_projectile(floor, i, MOB_FX)]) == wx
            ):
                var sp = Int(s[unsafe_offset=s_mob_projectile(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                var dy = Int(s[unsafe_offset=s_mob_projectile(floor, i, PROJ_FDIR_Y)])
                var dx = Int(s[unsafe_offset=s_mob_projectile(floor, i, PROJ_FDIR_X)])
                return _projectile_sprite_for(sp, dy, dx)
    for i in range(MAX_PLAYER_PROJECTILES):
        if s[unsafe_offset=s_player_projectile(floor, i, MOB_MASK)] > Float32(0.5):
            if (
                Int(s[unsafe_offset=s_player_projectile(floor, i, MOB_FY)]) == wy
                and Int(s[unsafe_offset=s_player_projectile(floor, i, MOB_FX)]) == wx
            ):
                var sp = Int(s[unsafe_offset=s_player_projectile(floor, i, MOB_TYPE_ID)])
                if sp < 0 or sp >= 8:
                    sp = 0
                var dy = Int(s[unsafe_offset=s_player_projectile(floor, i, PROJ_FDIR_Y)])
                var dx = Int(s[unsafe_offset=s_player_projectile(floor, i, PROJ_FDIR_X)])
                return _projectile_sprite_for(sp, dy, dx)
    return -1


# ============================================================================
# Inventory cell layout (4 rows × 10 cols — reference uses col=0..9):
#   (0,0)=HP  (0,1)=FD  (0,2)=DR  (0,3)=EN  (0,4)=MN  (0,5)= - (floor digit)
#   (0,6)= -  (0,7)=Hd  (0,8)=Bd  (0,9)=XP
#   (1,0)=W   (1,1)=Stn (1,2)=Cl  (1,3)=Ir  (1,4)=Dmd (1,5)=Sap (1,6)=Rby
#       (1,7)=Sapling   (1,8)=Bow (1,9)=DX
#   (2,0)=Wd  (2,1)=Stn (2,2)=Tor (2,3)=Bk  (2,4)=Fbl (2,5)=Ibl (2,6)=Arw
#       (2,7)=Pick      (2,8)=Sw  (2,9)=ST
#   (3,0..5)=Potions  (3,7)=Lg   (3,8)=Bt  (3,9)=INT
#
# Returns (sprite_idx, value); sprite_idx == -1 ⇒ empty cell.
# ============================================================================


@always_inline
def _row0_cell(
    s: Pointer[Float32, MutAnyOrigin], col: Int
) -> Tuple[Int, Int]:
    if col == 0:
        return (SPR_ICON_HEALTH, Int(s[unsafe_offset=s_intrinsic(INTRINSIC_HEALTH)]))
    elif col == 1:
        return (SPR_ICON_FOOD, Int(s[unsafe_offset=s_intrinsic(INTRINSIC_FOOD)]))
    elif col == 2:
        return (SPR_ICON_DRINK, Int(s[unsafe_offset=s_intrinsic(INTRINSIC_DRINK)]))
    elif col == 3:
        return (SPR_ICON_ENERGY, Int(s[unsafe_offset=s_intrinsic(INTRINSIC_ENERGY)]))
    elif col == 4:
        return (SPR_ICON_MANA, Int(s[unsafe_offset=s_intrinsic(INTRINSIC_MANA)]))
    elif col == 7:
        var head_tier = Int(s[unsafe_offset=s_inv(INV_ARMOUR_HEAD)])
        if head_tier <= 0:
            return (-1, 0)
        if head_tier > 2:
            head_tier = 2
        return (SPR_ARMOUR_BASE + 0 * 3 + head_tier, head_tier)
    elif col == 8:
        var body_tier = Int(s[unsafe_offset=s_inv(INV_ARMOUR_BODY)])
        if body_tier <= 0:
            return (-1, 0)
        if body_tier > 2:
            body_tier = 2
        return (SPR_ARMOUR_BASE + 1 * 3 + body_tier, body_tier)
    elif col == 9:
        return (SPR_ICON_XP, Int(s[unsafe_offset=s_attribute(ATTR_XP)]))
    return (-1, 0)


@always_inline
def _row1_cell(
    s: Pointer[Float32, MutAnyOrigin], col: Int
) -> Tuple[Int, Int]:
    # Materials row + sapling/bow.
    if col == 0:
        return (SPR_INV_LOG, Int(s[unsafe_offset=s_inv(INV_WOOD)]))
    elif col == 1:
        return (4, Int(s[unsafe_offset=s_inv(INV_STONE)]))  # stone sprite slot
    elif col == 2:
        return (8, Int(s[unsafe_offset=s_inv(INV_COAL)]))   # coal sprite slot
    elif col == 3:
        return (9, Int(s[unsafe_offset=s_inv(INV_IRON)]))   # iron
    elif col == 4:
        return (10, Int(s[unsafe_offset=s_inv(INV_DIAMOND)]))   # diamond
    elif col == 5:
        return (21, Int(s[unsafe_offset=s_inv(INV_SAPPHIRE)]))  # sapphire
    elif col == 6:
        return (22, Int(s[unsafe_offset=s_inv(INV_RUBY)]))      # ruby
    elif col == 7:
        return (15, Int(s[unsafe_offset=s_inv(INV_SAPLING)]))   # plant-young as sapling
    elif col == 8:
        return (SPR_BOW, Int(s[unsafe_offset=s_inv(INV_BOW)]))
    elif col == 9:
        return (SPR_ICON_DEX, Int(s[unsafe_offset=s_attribute(ATTR_DEXTERITY)]))
    return (-1, 0)


@always_inline
def _row2_cell(
    s: Pointer[Float32, MutAnyOrigin], col: Int
) -> Tuple[Int, Int]:
    # Tools + torches + books + arrows + spell unlocks.
    if col == 0:
        return (SPR_INV_TORCH, Int(s[unsafe_offset=s_inv(INV_TORCHES)]))
    elif col == 1:
        return (SPR_INV_BOOK, Int(s[unsafe_offset=s_inv(INV_BOOKS)]))
    elif col == 2:
        var leg_tier = Int(s[unsafe_offset=s_inv(INV_ARMOUR_LEGS)])
        if leg_tier <= 0:
            return (-1, 0)
        if leg_tier > 2:
            leg_tier = 2
        return (SPR_ARMOUR_BASE + 2 * 3 + leg_tier, leg_tier)
    elif col == 3:
        var feet_tier = Int(s[unsafe_offset=s_inv(INV_ARMOUR_FEET)])
        if feet_tier <= 0:
            return (-1, 0)
        if feet_tier > 2:
            feet_tier = 2
        return (SPR_ARMOUR_BASE + 3 * 3 + feet_tier, feet_tier)
    elif col == 4:
        var fb = (
            1 if s[unsafe_offset=s_learned_spell(SPELL_FIREBALL)] > Float32(0.5) else 0
        )
        if fb == 0:
            return (-1, 0)
        return (SPR_SPELL_FIREBALL, fb)
    elif col == 5:
        var ib = (
            1 if s[unsafe_offset=s_learned_spell(SPELL_ICEBALL)] > Float32(0.5) else 0
        )
        if ib == 0:
            return (-1, 0)
        return (SPR_SPELL_ICEBALL, ib)
    elif col == 6:
        return (SPR_ARROW_UP, Int(s[unsafe_offset=s_inv(INV_ARROWS)]))
    elif col == 7:
        var pick_tier = Int(s[unsafe_offset=s_inv(INV_PICKAXE)])
        if pick_tier < 0:
            pick_tier = 0
        if pick_tier > 4:
            pick_tier = 4
        return (SPR_PICKAXE_BASE + pick_tier, pick_tier)
    elif col == 8:
        var sword_tier = Int(s[unsafe_offset=s_inv(INV_SWORD)])
        if sword_tier < 0:
            sword_tier = 0
        if sword_tier > 4:
            sword_tier = 4
        return (SPR_SWORD_BASE + sword_tier, sword_tier)
    elif col == 9:
        return (SPR_ICON_STR, Int(s[unsafe_offset=s_attribute(ATTR_STRENGTH)]))
    return (-1, 0)


@always_inline
def _row3_cell(
    s: Pointer[Float32, MutAnyOrigin], col: Int
) -> Tuple[Int, Int]:
    # Potions in cols 0..5, INT in col 9.
    if col >= 0 and col < NUM_POTIONS:
        return (SPR_POTION_BASE + col, Int(s[unsafe_offset=s_inv(INV_POTIONS_BASE + col)]))
    if col == 9:
        return (SPR_ICON_INT, Int(s[unsafe_offset=s_attribute(ATTR_INTELLIGENCE)]))
    return (-1, 0)


@always_inline
def _inv_cell_sprite_and_value(
    s: Pointer[Float32, MutAnyOrigin],
    row: Int,
    col: Int,
) -> Tuple[Int, Int]:
    if row == 0:
        return _row0_cell(s, col)
    if row == 1:
        return _row1_cell(s, col)
    if row == 2:
        return _row2_cell(s, col)
    return _row3_cell(s, col)


# ============================================================================
# View-tile RGB — composites block + item + mob + player at (vy, vx).
# Called once per visible tile, then we expand to BPS×BPS pixels.
# ============================================================================


def _render_view_tile_rgb(
    s: Pointer[Float32, MutAnyOrigin],
    atlas: Pointer[Float32, MutAnyOrigin],
    floor: Int,
    py: Int,
    px: Int,
    vy: Int,
    vx: Int,
    ly: Int,
    lx: Int,
    is_boss_vulnerable: Bool,
    ladder_open: Bool,
) -> Tuple[Float32, Float32, Float32]:
    """Return the RGB for local pixel (ly, lx) inside view tile (vy, vx)."""
    var wy = py - VIEW_H // 2 + vy
    var wx = px - VIEW_W // 2 + vx
    var in_bounds = (wy >= 0 and wy < MAP_H and wx >= 0 and wx < MAP_W)

    var lit = True
    var block_id: Int = BLOCK_OUT_OF_BOUNDS
    var item_id: Int = 0
    if in_bounds:
        block_id = Int(s[unsafe_offset=s_map(floor, wy, wx)])
        item_id = Int(s[unsafe_offset=s_item_map(floor, wy, wx)])
        if floor != 0:
            var ltile = Float32(s[unsafe_offset=s_light_map(floor, wy, wx)])
            lit = ltile > Float32(0.05)
    if block_id == BLOCK_NECROMANCER and is_boss_vulnerable:
        block_id = BLOCK_NECROMANCER_VULNERABLE
    if item_id == ITEM_LADDER_DOWN and not ladder_open:
        item_id = ITEM_LADDER_DOWN_BLOCKED

    if not lit:
        # Dark tile — opaque black.
        return (Float32(0.0), Float32(0.0), Float32(0.0))

    # Base block.
    var base = _atlas_sample(atlas, block_id, ly, lx)
    var r = base[0]
    var g = base[1]
    var b = base[2]

    # Item overlay (torch / ladder).
    if item_id != 0:
        var ip = _atlas_sample(atlas, SPR_ITEM_BASE + item_id, ly, lx)
        var c = _composite(r, g, b, ip[0], ip[1], ip[2], ip[3])
        r = c[0]; g = c[1]; b = c[2]

    # Mob / projectile overlay.
    if in_bounds:
        var ms = _mob_at(s, floor, wy, wx)
        if ms >= 0:
            var mp = _atlas_sample(atlas, ms, ly, lx)
            var c = _composite(r, g, b, mp[0], mp[1], mp[2], mp[3])
            r = c[0]; g = c[1]; b = c[2]

    # Player at view center.
    if wy == py and wx == px:
        var pdir = Int(s[unsafe_offset=S_PLAYER_DIR])
        var sleeping = (
            s[unsafe_offset=s_intrinsic(INTRINSIC_IS_SLEEPING)] > Float32(0.5)
        )
        var ps_idx = _player_sprite_for(pdir, sleeping)
        var pp = _atlas_sample(atlas, ps_idx, ly, lx)
        var c = _composite(r, g, b, pp[0], pp[1], pp[2], pp[3])
        r = c[0]; g = c[1]; b = c[2]

    # Day/night dim — clamp to [0.3, 1.0] like Classic.
    var light = s[unsafe_offset=S_LIGHT_LEVEL]
    if floor != 0:
        light = Float32(1.0)
    if light < Float32(0.3):
        light = Float32(0.3)
    if light > Float32(1.0):
        light = Float32(1.0)
    return (r * light, g * light, b * light)


# ============================================================================
# Top-level per-pixel renderer
# ============================================================================


@always_inline
def _render_pixel_rgb_from_state(
    s: Pointer[Float32, MutAnyOrigin],
    atlas: Pointer[Float32, MutAnyOrigin],
    h: Int,
    w: Int,
) -> Tuple[Float32, Float32, Float32]:
    """GPU-friendly top-level: each thread owns one (env, pixel) and pulls
    floor / player_pos / boss-state / ladder-state out of the env's state
    slice itself. Slightly redundant CPU-side but keeps the kernel
    signature simple and avoids passing 5 extra args per thread."""
    var floor = Int(s[unsafe_offset=S_PLAYER_LEVEL])
    if floor < 0:
        floor = 0
    if floor >= NUM_FLOORS:
        floor = NUM_FLOORS - 1
    var py = Int(s[unsafe_offset=S_PLAYER_POS])
    var px = Int(s[unsafe_offset=S_PLAYER_POS + 1])
    var bossv = _is_boss_vulnerable(s)
    var ladder = _ladder_open(s, floor)
    return _render_pixel_rgb(
        s, atlas, floor, py, px, bossv, ladder, h, w,
    )


@always_inline
def _render_pixel_rgb(
    s: Pointer[Float32, MutAnyOrigin],
    atlas: Pointer[Float32, MutAnyOrigin],
    floor: Int,
    py: Int,
    px: Int,
    is_boss_vulnerable: Bool,
    ladder_open: Bool,
    h: Int,
    w: Int,
) -> Tuple[Float32, Float32, Float32]:
    if h < VIEW_PIX_H:
        var vy = h // BLOCK_PIXEL_SIZE
        var vx = w // BLOCK_PIXEL_SIZE
        var ly = h % BLOCK_PIXEL_SIZE
        var lx = w % BLOCK_PIXEL_SIZE
        return _render_view_tile_rgb(
            s, atlas, floor, py, px, vy, vx, ly, lx,
            is_boss_vulnerable, ladder_open,
        )
    # Inventory bar.
    var iy = h - VIEW_PIX_H
    var row = iy // BLOCK_PIXEL_SIZE
    var col = w // BLOCK_PIXEL_SIZE
    var ly = iy % BLOCK_PIXEL_SIZE
    var lx = w % BLOCK_PIXEL_SIZE

    var sv = _inv_cell_sprite_and_value(s, row, col)
    var sprite = sv[0]
    var value = sv[1]
    if sprite < 0 or value <= 0:
        return (Float32(0.05), Float32(0.05), Float32(0.08))
    var p = _atlas_sample(atlas, sprite, ly, lx)
    var bg_r = Float32(0.08)
    var bg_g = Float32(0.08)
    var bg_b = Float32(0.10)
    var c = _composite(bg_r, bg_g, bg_b, p[0], p[1], p[2], p[3])
    return (c[0], c[1], c[2])


# ============================================================================
# CraftaxFullPixelEnv
# ============================================================================


struct CraftaxFullPixelEnv[DTYPE: DType = DType.float32](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Craftax-Full with 3×130×110 RGB sprite-based pixel obs.

    Channel-first layout (C, H, W) flat row-major. Single frame, no stack —
    matches the reference `render_craftax_pixels` at `BLOCK_PIXEL_SIZE_AGENT`.

    Physics + human-playable rendering delegate to `CraftaxFullEnv`. GPU
    kernels: reset/selective_reset/step/extract_obs all wired (#42).
    """

    comptime dtype = Self.DTYPE
    comptime StateType = CraftaxFullState
    comptime ActionType = CraftaxFullAction

    comptime STATE_SIZE: Int = STATE_SIZE
    comptime OBS_DIM: Int = PIXEL_OBS_DIM
    comptime NUM_ACTIONS: Int = NUM_ACTIONS
    comptime STEP_WS_SHARED: Int = ATLAS_FLOATS
    comptime STEP_WS_PER_ENV: Int = 0
    comptime TPB: Int = 256

    var inner: CraftaxFullEnv[Self.DTYPE]
    var _atlas: Pointer[Float32, MutUntrackedOrigin]
    var _atlas_loaded: Bool

    def __init__(out self):
        self.inner = CraftaxFullEnv[Self.DTYPE]()
        self._atlas_loaded = False
        try:
            self._atlas = build_agent_atlas(ASSET_DIR, BLOCK_PIXEL_SIZE)
            self._atlas_loaded = True
        except e:
            print("Craftax-Full pixel env: atlas load failed (",
                  String(e), ")")
            self._atlas = alloc[Float32]({count = ATLAS_FLOATS}).unsafe_leak()
            for i in range(ATLAS_FLOATS):
                self._atlas[unsafe_offset=i] = Float32(0.0)

    def __deinit__(deinit self):
        if Int(self._atlas) != 0:
            self._atlas.unsafe_free()

    # ------------------------------------------------------------------
    # CPU render
    # ------------------------------------------------------------------

    @always_inline
    def _render_current(
        self, mut obs: Pointer[Scalar[Self.DTYPE], MutAnyOrigin]
    ):
        var state_ptr = rebind[Pointer[Float32, MutAnyOrigin]](
            self.inner.state.unsafe_ptr().unsafe_bitcast[Float32]()
        )
        var atlas = self._atlas
        var floor = Int(state_ptr[unsafe_offset=S_PLAYER_LEVEL])
        if floor < 0:
            floor = 0
        if floor >= NUM_FLOORS:
            floor = NUM_FLOORS - 1
        var py = Int(state_ptr[unsafe_offset=S_PLAYER_POS])
        var px = Int(state_ptr[unsafe_offset=S_PLAYER_POS + 1])
        var bossv = _is_boss_vulnerable(state_ptr)
        var ladder = _ladder_open(state_ptr, floor)

        comptime HW = OBS_PIX_H * OBS_PIX_W
        for h in range(OBS_PIX_H):
            for w in range(OBS_PIX_W):
                var rgb = _render_pixel_rgb(
                    state_ptr,
                    atlas.as_unsafe_any_origin(),
                    floor,
                    py,
                    px,
                    bossv,
                    ladder,
                    h,
                    w,
                )
                obs[unsafe_offset=0 * HW + h * OBS_PIX_W + w] = Scalar[Self.DTYPE](rgb[0])
                obs[unsafe_offset=1 * HW + h * OBS_PIX_W + w] = Scalar[Self.DTYPE](rgb[1])
                obs[unsafe_offset=2 * HW + h * OBS_PIX_W + w] = Scalar[Self.DTYPE](rgb[2])

    # ------------------------------------------------------------------
    # Env trait — delegate physics to inner
    # ------------------------------------------------------------------

    def reset(mut self) -> CraftaxFullState:
        return self.inner.reset()

    def reset_with_seed(mut self, seed: UInt64) -> CraftaxFullState:
        return self.inner.reset_with_seed(seed)

    def step(
        mut self, action: CraftaxFullAction, verbose: Bool = False
    ) -> Tuple[CraftaxFullState, Scalar[Self.DTYPE], Bool]:
        return self.inner.step(action, verbose)

    def get_state(mut self) -> CraftaxFullState:
        return self.inner.get_state()

    def close(mut self):
        self.inner.close()

    def action_from_index(self, action_idx: Int) -> CraftaxFullAction:
        return self.inner.action_from_index(action_idx)

    def num_actions(self) -> Int:
        return NUM_ACTIONS

    def obs_dim(self) -> Int:
        return PIXEL_OBS_DIM

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: CraftaxFullState) -> Int:
        return state.index

    # ------------------------------------------------------------------
    # BoxDiscreteActionEnv (obs path)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # RenderableEnv — delegate to inner
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # GPU kernels (#42) — physics delegates to inner, render is local.
    # ------------------------------------------------------------------

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Delegate to the symbolic env's reset — pixel env shares the
        underlying state buffer."""
        CraftaxFullEnv[Self.DTYPE].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, rng_seed=rng_seed,
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
        CraftaxFullEnv[Self.DTYPE].selective_reset_kernel_gpu[
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
        """Copy the (already-built) CPU atlas into the shared region of the
        GPU workspace. Layout: `[shared(ATLAS_FLOATS) | per-env padding(0)]`."""
        ctx.enqueue_copy(workspace_buf, self._atlas)

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype],) raises:
        """Static fallback: rebuild the atlas on host, upload, free."""
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
    def _render_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[gpu_dtype],
        atlas_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """One thread per output pixel — writes (R, G, B) for that pixel."""
        comptime PIX_TOTAL = BATCH_SIZE * OBS_PIX_H * OBS_PIX_W
        comptime PIX_BLOCKS = (PIX_TOTAL + Self.TPB - 1) // Self.TPB
        # Borrowed immutably, so the pointer now carries an immutable
        # origin; the kernel ABI declares `MutAnyOrigin`. Device
        # allocations are outside origin tracking, so this only restores
        # the pre-nightly typing.
        var states_ptr = states_buf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
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
            var rgb = _render_pixel_rgb_from_state(state, atlas_ptr, h, w)
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
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Cold extract used after reset. Re-uploads the atlas to a temp
        buffer — training paths use step_kernel_gpu with workspace_ptr."""
        var atlas_buf = ctx.enqueue_create_buffer[gpu_dtype](ATLAS_FLOATS)
        var host = build_agent_atlas(ASSET_DIR, BLOCK_PIXEL_SIZE)
        ctx.enqueue_copy(atlas_buf, host)
        ctx.synchronize()
        host.unsafe_free()

        Self._render_kernel[BATCH_SIZE, STATE_SIZE](
            ctx, states_buf, atlas_buf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True](), obs_buf,
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
        """Physics step → render the pixel obs. Atlas lives in the shared
        region of the workspace (offset 0)."""
        var states_ptr = states_buf.unsafe_ptr()
        var actions_ptr = actions_buf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
        var rewards_ptr = rewards_buf.unsafe_ptr()
        var dones_ptr = dones_buf.unsafe_ptr()
        var terminated_ptr = terminated_buf.unsafe_ptr()

        comptime PHYS_BLOCKS = (
            BATCH_SIZE + CraftaxFullEnv[Self.DTYPE].TPB - 1
        ) // CraftaxFullEnv[Self.DTYPE].TPB
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
            block_dim=(CraftaxFullEnv[Self.DTYPE].TPB,),
        )

        # Render pass.
        if workspace_ptr:
            var ws_ptr = workspace_ptr.value()
            Self._render_kernel[BATCH_SIZE, STATE_SIZE](
                ctx, states_buf, ws_ptr, obs_buf,
            )
        else:
            # Fallback path: upload atlas to a temp buffer for this call.
            var atlas_buf = ctx.enqueue_create_buffer[gpu_dtype](ATLAS_FLOATS)
            var host = build_agent_atlas(ASSET_DIR, BLOCK_PIXEL_SIZE)
            ctx.enqueue_copy(atlas_buf, host)
            ctx.synchronize()
            host.unsafe_free()
            Self._render_kernel[BATCH_SIZE, STATE_SIZE](
                ctx, states_buf, atlas_buf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True](), obs_buf,
            )
