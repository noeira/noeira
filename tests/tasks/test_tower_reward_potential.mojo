"""The potential-based reward mode — `META_IDX_REWARD_MODE == 1` — on the tower.

    pixi run mojo run -I . tests/tasks/test_tower_reward_potential.mojo

`So101FamilyConfig.compute_reward_and_done_gpu` pays, in mode 1, the CHANGE
of a staged potential (`noeira-docs/SO101_PIXEL_RL_PLAN.md`,
so101-nexus's fix for dwelling rewards):

    Phi = w_goal goal + w_reach max(reach, H) + GRASP_W max(grasp, H)
          + CLOSE_W max(close, H)          H = the goal holds
    r   = Phi - Phi_prev                  (0 on an episode's first step)
    r   = W (the weights' sum) while the goal holds, + the bonus once

This is gated on the HOST through `family_reward_host` (the kernel's own
function), against the LEGACY mode as the independent oracle: with the jaw
away from the brick and the goal not holding, there is no grasp and no
closing term, so `Phi` IS the legacy reward. Each state is scored in both
modes (legacy first — it writes no episode state), so

1. the first step of an episode pays exactly 0 and records `Phi`;
2. every later step pays exactly `L_t - L_{t-1}`, and the steps telescope;
3. scoring the SAME state again pays exactly 0 — the dwelling exploit, closed;
4. the brick lifted (the goal holds) pays `W + bonus` once, then `W`, and
   the paid flag is set;
5. `reset()` clears the episode state: the next first step pays 0 again;
6. mode 0 with the new words present is the legacy reward (the words are
   inert) — and `test_tower_host_reward.mojo` still gates the legacy values.
"""

from std.math import abs
from std.testing import assert_almost_equal, assert_equal, assert_true
from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.gpu.constants import (
    META_IDX_REWARD_MODE, META_IDX_SUCCESS_BONUS, META_IDX_PHI_PREV,
    META_IDX_EPISODE_FLAGS, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.host_reward import family_reward_host
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.shaping import reward_mode_words
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime DTYPE = DType.float64
comptime CFG = So101TowerConfig
comptime E = Phyics3dEnv[So101TowerModel, CFG, DTYPE, False]
comptime ACT = E.ACTION_DIM
comptime FAMILY = "so101_tower"
comptime TASK = "so101_tower_lift_brick"
comptime TOL = 1e-12
comptime BONUS = 5.0


def _score(mut env: E, mode: Int, step: Int) raises -> Tuple[Float64, Bool]:
    env.d.meta.data[META_IDX_REWARD_MODE] = Scalar[DTYPE](mode)
    var zero = List[Float64](length=ACT, fill=0.0)
    var r = family_reward_host[CFG, DTYPE, E.MD, ACT](
        env.d, env.mf, zero, step, env.frame_skip, So101TowerModel.TIMESTEP
    )
    return (Float64(r[0]), r[1])


def _setup(mut env: E) raises:
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
        env.mf.curriculum.data[i] = Scalar[DTYPE](cw[i])
    var mw = task_meta_words(
        String(TASK), String(FAMILY), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    for k in range(len(mw[0])):
        env.d.meta.data[mw[0][k]] = Scalar[DTYPE](mw[1][k])
    var rw = reward_mode_words(True, BONUS)
    env.d.meta.data[META_IDX_SUCCESS_BONUS] = Scalar[DTYPE](rw[1])


def main() raises:
    print("=" * 66)
    print("potential-based reward mode, on", TASK)
    print("=" * 66)
    var ctx = DeviceContext()
    var env = E(ctx)
    _ = env.reset()
    _setup(env)
    var q0 = posed_qpos[So101TowerPlacement](
        String(TASK), String(FAMILY), CFG.SLOT_RADIUS
    )
    var v0 = List[Float64](length=So101TowerModel.NV, fill=0.0)
    _ = env.obs_at(q0, v0)

    # ── 1. the first step of the episode: 0, and Phi recorded ────────────
    var l0 = _score(env, 0, 0)
    assert_equal(Float64(env.d.meta.data[META_IDX_EPISODE_FLAGS]), 0.0,
                 "mode 0 writes no episode state")
    var p0 = _score(env, 1, 0)
    print("  step 0: legacy", l0[0], " potential", p0[0])
    assert_equal(p0[0], 0.0, "the first step of an episode pays 0")
    assert_equal(Float64(env.d.meta.data[META_IDX_PHI_PREV]), l0[0],
                 "Phi recorded (Phi == the legacy reward here)")
    assert_equal(Float64(env.d.meta.data[META_IDX_EPISODE_FLAGS]), 1.0,
                 "the Phi-set bit, no bonus")

    # ── 3. the same state again: exactly 0 (the dwell) ──────────────────
    var again = _score(env, 1, 0)
    assert_equal(again[0], 0.0, "scoring the same state again pays 0")

    # ── 2. steps: each pays L_t - L_{t-1}, and the sum telescopes ────────
    var a = ContAction[ACT]()
    var prev_l = l0[0]
    var sum_p = 0.0
    var moved = 0.0
    for t in range(8):
        _ = env.step(a)
        var l = _score(env, 0, t + 1)
        var p = _score(env, 1, t + 1)
        assert_true(not l[1], "no lift by itself")
        assert_almost_equal(p[0], l[0] - prev_l, atol=TOL,
                            msg="step " + String(t + 1) + " pays L_t - L_{t-1}")
        moved += abs(l[0] - prev_l)
        sum_p += p[0]
        prev_l = l[0]
    assert_almost_equal(sum_p, prev_l - l0[0], atol=1e-10,
                        msg="the deltas telescope to L_T - L_0")
    print("  8 steps: sum of deltas", sum_p, "= L_T - L_0", prev_l - l0[0],
          " (total |dL|", moved, ")")
    assert_true(moved > 0.0, "the settling steps moved the reward at all"
                             " (else items 2 and 3 are one test)")

    # ── 4. the goal holds: W + bonus once, then W ────────────────────────
    var q = List[Float64]()
    for i in range(So101TowerModel.NQ):
        q.append(Float64(env.d.qpos.data[i]))
    for j in range(So101TowerPlacement.N_FREE):
        var adr = So101TowerPlacement.free_qadr(j)
        q[adr + 2] += 0.20
    _ = env.obs_at(q, v0)
    var w = CFG.SHAPE_W_GOAL + CFG.SHAPE_W_REACH + CFG.GRASP_W + CFG.CLOSE_W
    var h1 = _score(env, 1, 9)
    var h2 = _score(env, 1, 10)
    print("  goal holds: first", h1[0], " then", h2[0], " (W =", w, ", bonus",
          BONUS, ")")
    assert_true(h1[1] and h2[1], "Above holds with the brick 20 cm up")
    assert_almost_equal(h1[0], w + BONUS, atol=TOL, msg="W + bonus, once")
    assert_almost_equal(h2[0], w, atol=TOL, msg="then W per step")
    assert_equal(Float64(env.d.meta.data[META_IDX_EPISODE_FLAGS]), 3.0,
                 "the bonus is marked paid (and Phi set)")

    # ── 5. reset clears the episode state ────────────────────────────────
    _ = env.reset()
    assert_equal(Float64(env.d.meta.data[META_IDX_PHI_PREV]), 0.0,
                 "reset zeroes Phi_prev")
    assert_equal(Float64(env.d.meta.data[META_IDX_EPISODE_FLAGS]), 0.0,
                 "reset zeroes the episode flags")
    _setup(env)
    _ = env.obs_at(q0, v0)
    var r0 = _score(env, 1, 0)
    assert_equal(r0[0], 0.0, "after a reset the first step pays 0 again")

    # ── 6. mode 0 with the words present is the legacy reward ────────────
    var l_again = _score(env, 0, 0)
    assert_almost_equal(l_again[0], l0[0], atol=TOL,
                        msg="mode 0 ignores the reward block")
    print("TOWER POTENTIAL REWARD OK")
