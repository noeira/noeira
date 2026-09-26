"""`shaping.near_transport_shortfall` — the lift-free `Near` distance of the
potential-based reward.

    pixi run mojo run -I . tests/tasks/test_near_transport.mojo

The potential-based mode measures a single `Near(a, b, r)` goal as
`max(|horizontal|, |vertical|) - r` instead of the 3D shortfall, because the
3D one pays NEGATIVE for lifting the brick off the desk — the move
`so101_tower_cube_in_bowl` needs to clear the bowl's rim. Checked here on the
function itself (the reward hook's use of it is checked through the host
reward in `test_tower_reward_potential.mojo`):

1. it is never larger than the 3D shortfall (so it reaches 0 no later than
   the predicate), over random offsets;
2. lifting is FREE while the horizontal gap exceeds the height gap;
3. along an ideal pick-carry-lower trajectory it never increases;
4. it is 0 at a brick resting in the bowl (the predicate holds);
5. the brick pushed against the bowl's outside ranks WORSE than the brick
   held over the bowl — the local optimum the 3D measure created;
6. THE HOOK USES IT: on `so101_tower_cube_in_bowl`, through
   `family_reward_host` (the kernel's own function), with the jaw away from
   the brick (no grasp, no close term) the mode-1 potential minus the legacy
   reward at the SAME state is exactly `w_goal (tol(transport) - tol(3D))`,
   at rest and with the brick lifted 6 cm; the lift leaves the potential's
   goal term unchanged where the legacy one drops; on the lift task (`Above`)
   the difference stays 0.
"""

from std.math import sqrt, abs
from std.random import random_float64, seed
from std.testing import assert_true, assert_equal, assert_almost_equal
from noeira.tasks.predicates import OP_NEAR

from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.envs.dm_control.rewards import tolerance
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.gpu.constants import (
    META_IDX_REWARD_MODE, META_IDX_SUCCESS_BONUS, META_IDX_PHI_PREV,
    META_IDX_EPISODE_FLAGS, META_IDX_TASK_PARAM_0, META_IDX_SHAPE_W_GOAL,
    META_IDX_GOAL_MARGIN, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.host_reward import family_reward_host
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.shaping import near_transport_shortfall
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime CFG = So101TowerConfig
comptime E = Phyics3dEnv[So101TowerModel, CFG, DType.float64, False]
comptime ACT = E.ACTION_DIM
comptime FAMILY = "so101_tower"

comptime DT = DType.float64
comptime R = 0.045


def sf(ex: Float64, ey: Float64, ez: Float64) -> Float64:
    return Float64(near_transport_shortfall[DT](ex, ey, ez, R))


def _setup(mut env: E, task: String) raises:
    var f = load_family("noeira/tasks/families/so101_tower.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for i in range(MODEL_CURRICULUM_SIZE):
        env.mf.curriculum.data[i] = Float64(cw[i])
    var mw = task_meta_words(
        task, String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    for k in range(len(mw[0])):
        env.d.meta.data[mw[0][k]] = Float64(mw[1][k])
    env.d.meta.data[META_IDX_SUCCESS_BONUS] = 0.0


def _legacy(mut env: E) raises -> Float64:
    env.d.meta.data[META_IDX_REWARD_MODE] = 0.0
    var zero = List[Float64](length=ACT, fill=0.0)
    var r = family_reward_host[CFG, DType.float64, E.MD, ACT](
        env.d, env.mf, zero, 0, env.frame_skip, So101TowerModel.TIMESTEP
    )
    assert_true(not r[1], "the goal does not hold in these states")
    return Float64(r[0])


def _phi(mut env: E) raises -> Float64:
    """Mode 1's potential at the current state: a first step records it."""
    env.d.meta.data[META_IDX_REWARD_MODE] = 1.0
    env.d.meta.data[META_IDX_EPISODE_FLAGS] = 0.0
    var zero = List[Float64](length=ACT, fill=0.0)
    _ = family_reward_host[CFG, DType.float64, E.MD, ACT](
        env.d, env.mf, zero, 0, env.frame_skip, So101TowerModel.TIMESTEP
    )
    return Float64(env.d.meta.data[META_IDX_PHI_PREV])


def _goal_terms(env: E) -> Tuple[Float64, Float64]:
    """(tol(3D shortfall), tol(transport shortfall)) of term 0, recomputed."""
    var a = Int(env.d.meta.data[META_IDX_TASK_PARAM_0 + 1])
    var b = Int(env.d.meta.data[META_IDX_TASK_PARAM_0 + 2])
    var rad = Float64(env.d.meta.data[META_IDX_TASK_PARAM_0 + 3])
    var m = Float64(env.d.meta.data[META_IDX_GOAL_MARGIN])
    var ex = Float64(env.d.xpos.data[a * 3] - env.d.xpos.data[b * 3])
    var ey = Float64(env.d.xpos.data[a * 3 + 1] - env.d.xpos.data[b * 3 + 1])
    var ez = Float64(env.d.xpos.data[a * 3 + 2] - env.d.xpos.data[b * 3 + 2])
    var d3 = sqrt(ex * ex + ey * ey + ez * ez) - rad
    if d3 < 0:
        d3 = 0
    var dt = Float64(near_transport_shortfall[DT](ex, ey, ez, rad))
    var g3 = Float64(tolerance[DTYPE=DT](d3, 0.0, CFG.GOAL_RADIUS, m))
    var gt = Float64(tolerance[DTYPE=DT](dt, 0.0, CFG.GOAL_RADIUS, m))
    return (g3, gt)


def _hook_section() raises:
    var ctx = DeviceContext()
    var env = E(ctx)
    _ = env.reset()
    comptime TASK = "so101_tower_cube_in_bowl"
    _setup(env, String(TASK))
    assert_equal(Int(env.d.meta.data[META_IDX_TASK_PARAM_0]), OP_NEAR,
                 "cube_in_bowl is a Near goal")
    var wg = Float64(env.d.meta.data[META_IDX_SHAPE_W_GOAL])
    var q0 = posed_qpos[So101TowerPlacement](
        String(TASK), String(FAMILY), CFG.SLOT_RADIUS
    )
    var v0 = List[Float64](length=So101TowerModel.NV, fill=0.0)
    _ = env.obs_at(q0, v0)
    var brick = Int(env.d.meta.data[META_IDX_TASK_PARAM_0 + 1])
    # the free joint that carries the brick
    var z_before = Float64(env.d.xpos.data[brick * 3 + 2])
    var jb = -1
    for j in range(So101TowerPlacement.N_FREE):
        var q = q0.copy()
        q[So101TowerPlacement.free_qadr(j) + 2] += 0.06
        _ = env.obs_at(q, v0)
        if abs(Float64(env.d.xpos.data[brick * 3 + 2]) - z_before - 0.06) < 1e-9:
            jb = j
    assert_true(jb >= 0, "found the brick's free joint")

    var at_rest = Tuple[Float64, Float64](0.0, 0.0)
    for k in range(2):
        var q = q0.copy()
        q[So101TowerPlacement.free_qadr(jb) + 2] += 0.06 * Float64(k)
        _ = env.obs_at(q, v0)
        var l = _legacy(env)
        var p = _phi(env)
        var g = _goal_terms(env)
        assert_almost_equal(p - l, wg * (g[1] - g[0]), atol=1e-12,
                            msg="Phi - L = w_goal (tol(transport) - tol(3D))")
        print("  6. cube_in_bowl", "lifted 6 cm" if k == 1 else "at rest    ",
              ": 3D goal", g[0], " transport goal", g[1], " Phi - L", p - l)
        if k == 0:
            at_rest = g
        else:
            assert_true(g[0] < at_rest[0], "the 3D goal term drops on the lift")
            assert_almost_equal(g[1], at_rest[1], atol=1e-12,
                                msg="the transport goal term does not")
    # the lift task is `Above`: untouched
    _ = env.reset()
    comptime LIFT = "so101_tower_lift_brick"
    _setup(env, String(LIFT))
    var ql = posed_qpos[So101TowerPlacement](
        String(LIFT), String(FAMILY), CFG.SLOT_RADIUS
    )
    _ = env.obs_at(ql, v0)
    var l = _legacy(env)
    assert_equal(_phi(env), l, "an Above goal's potential is the legacy reward")
    print("  6. lift_brick (Above): Phi == L ==", l)


def main() raises:
    seed(3)
    # 1. never above the 3D shortfall
    for _ in range(10000):
        var ex = random_float64(-0.4, 0.4)
        var ey = random_float64(-0.4, 0.4)
        var ez = random_float64(-0.2, 0.2)
        var d3 = sqrt(ex * ex + ey * ey + ez * ez) - R
        if d3 < 0:
            d3 = 0
        assert_true(sf(ex, ey, ez) <= d3 + 1e-15, "transport <= 3D shortfall")
    print("  1. transport shortfall <= 3D shortfall on 10000 offsets")

    # 2. lifting is free while |horizontal| > |vertical|
    var rest = sf(0.20, 0.0, 0.0125)
    var up = sf(0.20, 0.0, 0.0725)
    assert_equal(rest, up, "a 6 cm lift at 20 cm is free")
    var d3r = sqrt(0.2 * 0.2 + 0.0125 * 0.0125) - R
    var d3u = sqrt(0.2 * 0.2 + 0.0725 * 0.0725) - R
    assert_true(d3u > d3r, "the 3D shortfall grows on the lift (the trap)")
    print("  2. lift at 20 cm:", rest, "->", up, "(3D:", d3r, "->", d3u, ")")

    # 3. monotone along reach-lift-carry-lower
    var prev = 1e9
    var n = 0
    # lift from the desk at 20 cm
    for k in range(20):
        var dz = 0.0125 + 0.06 * Float64(k) / 19.0
        var v = sf(0.20, 0.0, dz)
        assert_true(v <= prev + 1e-15, "lift phase never increases")
        prev = v
        n += 1
    # carry toward the bowl at 7.25 cm
    for k in range(40):
        var dx = 0.20 * (1.0 - Float64(k) / 39.0)
        var v = sf(dx, 0.0, 0.0725)
        assert_true(v <= prev + 1e-15, "carry phase never increases")
        prev = v
        n += 1
    # lower into the bowl
    for k in range(20):
        var dz = 0.0725 - (0.0725 - 0.018) * Float64(k) / 19.0
        var v = sf(0.0, 0.0, dz)
        assert_true(v <= prev + 1e-15, "lower phase never increases")
        prev = v
        n += 1
    print("  3. non-increasing over", n, "points of the ideal trajectory")

    # 4. zero at rest in the bowl
    assert_equal(sf(0.02, 0.0, 0.018), 0.0, "0 with the brick in the bowl")
    print("  4. 0 at a brick resting in the bowl")

    # 5. pushed against the outside vs held over the bowl
    var pushed = sf(0.078, 0.0, 0.0125)
    var held = sf(0.0, 0.0, 0.06)
    assert_true(held < pushed, "held over the bowl beats pushed against it")
    print("  5. pushed against the rim", pushed, "> held over the bowl", held)

    # 6. the reward hook uses it, on cube_in_bowl only
    _hook_section()
    print("NEAR TRANSPORT OK")
