"""The recorder's reward IS the trainer's — `tasks/host_reward.mojo` gate.

    pixi run mojo run -I . tests/tasks/test_tower_host_reward.mojo

`family_reward_host` wraps a CPU env's `Data`/`Model` columns in host views
and calls `So101TowerConfig.compute_reward_and_done_gpu` — the kernel's own
function. What can still be wrong is the WIRING: a view over the wrong column
(site poses where body poses go, a contact row width off by one) evaluates
without an error and pays a plausible number. So this recomputes the two
shaped terms from an INDEPENDENT source — the observation's goal words,
written by `write_task_obs_host` and gated word-for-word against the GPU
hook by `test_active_mask` — and demands equality to float64 rounding:

    goal  : `Above(brick, desk, m)` shortfall = max(0, (desk_z + m) - brick_z)
            = max(0, obs[target - subject]_z + m), `m` read off the TAPE
    reach : |subject - gripper| from the same words

at the posed reset (arm folded, brick on the desk: no contact rung, no
closing bonus), then along a few zero-action steps (the props settle, the
arm holds), then with the brick LIFTED by hand — the predicate must flip,
`META_IDX_GOAL_HELD` must be written, and the goal term must saturate.

⚠ THE GRASP RUNG AND THE CLOSING BONUS ARE NOT EXERCISED HERE — they need
the jaw at the brick, which no posed state reaches without a policy. They
are the kernel's own code (unchanged by the wrapper); the smoke run of the
recorder with `--policy` on a parked checkpoint is where their values are
read against the training log's `mean_reward`.
"""

from std.math import sqrt
from std.testing import assert_almost_equal, assert_equal, assert_true
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from noeira.core.cont_action import ContAction
from noeira.envs.dm_control.rewards import (
    tolerance, SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN,
)
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.gpu.constants import (
    META_IDX_GOAL_HELD, META_IDX_TASK_PARAM_0, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.host_reward import family_reward_host
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime DTYPE = DType.float64
comptime CFG = So101TowerConfig
comptime E = Phyics3dEnv[So101TowerModel, CFG, DTYPE, False]
comptime ACT = E.ACTION_DIM
comptime GB = CFG.OBS_GOAL_BASE
comptime FAMILY = "so101_tower"
comptime TASK = "so101_tower_lift_brick"
comptime TOL = 1e-9


def _expected_from_obs(ref obs: List[Float64], above_margin: Float64) -> Float64:
    """The two shaped terms, from the goal words alone."""
    var rx = obs[GB + 3]
    var ry = obs[GB + 4]
    var rz = obs[GB + 5]
    var reach = sqrt(rx * rx + ry * ry + rz * rz)
    var short = obs[GB + 8] + above_margin
    if short < 0.0:
        short = 0.0
    var r = CFG.SHAPE_W_GOAL * Float64(tolerance[
        SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
    ](
        Scalar[DTYPE](short), Scalar[DTYPE](0), Scalar[DTYPE](CFG.GOAL_RADIUS),
        Scalar[DTYPE](CFG.GOAL_MARGIN),
    ))
    r += CFG.SHAPE_W_REACH * Float64(tolerance[
        SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
    ](
        Scalar[DTYPE](reach), Scalar[DTYPE](0), Scalar[DTYPE](CFG.REACH_RADIUS),
        Scalar[DTYPE](CFG.REACH_MARGIN),
    ))
    return r


def _reach(ref obs: List[Float64]) -> Float64:
    var rx = obs[GB + 3]
    var ry = obs[GB + 4]
    var rz = obs[GB + 5]
    return sqrt(rx * rx + ry * ry + rz * rz)


def _run() raises:
    print("=" * 66)
    print("host reward == the family's GPU hook, on", TASK)
    print("=" * 66)
    var ctx = DeviceContext()
    var env = E(ctx)

    # the region table (the driver's upload) and the task's meta words
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
    _ = env.reset()
    for k in range(len(mw[0])):
        env.d.meta.data[mw[0][k]] = Scalar[DTYPE](mw[1][k])
    var q0 = posed_qpos[So101TowerPlacement](
        String(TASK), String(FAMILY), CFG.SLOT_RADIUS
    )
    var v0 = List[Float64](length=So101TowerModel.NV, fill=0.0)
    var s0 = env.obs_at(q0, v0)
    var obs = List[Float64]()
    for i in range(E.OBS_DIM):
        obs.append(s0.data[i])
    # the tape's third word of term 0 is `Above`'s margin
    var above_m = Float64(env.d.meta.data[META_IDX_TASK_PARAM_0 + 3])
    print("  Above margin (tape):", above_m, " reach at reset:", _reach(obs))
    assert_true(
        _reach(obs) > CFG.CLOSE_RADIUS,
        "the posed reset must start with the jaw away from the brick (no"
        " closing bonus in the expectation)",
    )
    var zero = List[Float64](length=ACT, fill=0.0)

    # ── 1. at the posed reset ────────────────────────────────────────────
    var rd = family_reward_host[CFG, DTYPE, E.MD, ACT](
        env.d, env.mf, zero, 0, env.frame_skip, So101TowerModel.TIMESTEP
    )
    var exp0 = _expected_from_obs(obs, above_m)
    print("  reset: host reward", Float64(rd[0]), " expected", exp0,
          " holds", rd[1])
    assert_almost_equal(Float64(rd[0]), exp0, atol=TOL,
                        msg="reward at the posed reset")
    assert_true(not rd[1], "the goal does not hold at reset")
    assert_equal(Float64(env.d.meta.data[META_IDX_GOAL_HELD]), 0.0,
                 "GOAL_HELD written as 0")
    var r0 = Float64(rd[0])

    # ── 2. along zero-action steps — the props settle, the arm stays ─────
    var a = ContAction[ACT]()
    for t in range(5):
        var out = env.step(a)
        var ob = List[Float64]()
        for i in range(E.OBS_DIM):
            ob.append(out[0].data[i])
        var r = family_reward_host[CFG, DTYPE, E.MD, ACT](
            env.d, env.mf, zero, t + 1, env.frame_skip,
            So101TowerModel.TIMESTEP,
        )
        var ex = _expected_from_obs(ob, above_m)
        assert_true(
            _reach(ob) > CFG.CLOSE_RADIUS, "still no closing bonus expected"
        )
        assert_almost_equal(Float64(r[0]), ex, atol=TOL,
                            msg="reward at step " + String(t + 1))
        assert_true(not r[1], "no lift happened by itself")
    print("  5 zero-action steps: host reward == goal-word recomputation")

    # ── 3. the brick lifted by hand: the predicate flips ─────────────────
    var q = List[Float64]()
    for i in range(So101TowerModel.NQ):
        q.append(Float64(env.d.qpos.data[i]))
    # every free slot 20 cm up — `Above(brick, desk, m)` then holds
    for j in range(So101TowerPlacement.N_FREE):
        var adr = So101TowerPlacement.free_qadr(j)
        q[adr + 2] += 0.20
    var s_up = env.obs_at(q, v0)
    var ob_up = List[Float64]()
    for i in range(E.OBS_DIM):
        ob_up.append(s_up.data[i])
    var r_up = family_reward_host[CFG, DTYPE, E.MD, ACT](
        env.d, env.mf, zero, 6, env.frame_skip, So101TowerModel.TIMESTEP
    )
    var ex_up = _expected_from_obs(ob_up, above_m)
    print("  lifted: host reward", Float64(r_up[0]), " expected", ex_up,
          " holds", r_up[1])
    assert_true(r_up[1], "Above holds with the brick 20 cm up")
    assert_equal(Float64(env.d.meta.data[META_IDX_GOAL_HELD]), 1.0,
                 "GOAL_HELD written as 1")
    assert_almost_equal(Float64(r_up[0]), ex_up, atol=TOL,
                        msg="reward with the brick lifted")
    assert_true(Float64(r_up[0]) > r0, "the lift pays more than the reset")
    # the goal term saturates at its weight when the shortfall is 0
    var goal_only = ex_up - CFG.SHAPE_W_REACH * Float64(tolerance[
        SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DTYPE
    ](
        Scalar[DTYPE](_reach(ob_up)), Scalar[DTYPE](0),
        Scalar[DTYPE](CFG.REACH_RADIUS), Scalar[DTYPE](CFG.REACH_MARGIN),
    ))
    assert_almost_equal(goal_only, CFG.SHAPE_W_GOAL, atol=TOL,
                        msg="goal term saturates when the goal holds")
    print("TOWER HOST REWARD OK")


def main() raises:
    # The env is built on a DeviceContext, so there is nothing to run on a
    # host with no accelerator (the CI runner).
    comptime if has_accelerator():
        _run()
    else:
        print(
            "test_tower_host_reward: SKIPPED (no accelerator;"
            " run with -e apple / -e nvidia)"
        )
