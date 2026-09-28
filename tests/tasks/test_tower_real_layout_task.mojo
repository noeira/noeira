"""`so101_tower_lift_real_layout` draws EXACTLY cube-in-bowl's start states.

    pixi run mojo run -I . tests/tasks/test_tower_real_layout_task.mojo

The lift stage of the cube-in-bowl curriculum is only useful if its resets
are cube-in-bowl's (`tasks/so101_tower_lift_real_layout.task`). The device
draws a lane's reset from the task's `meta` words, so: every word
`task_meta_words` writes for the two tasks is equal, except the goal tape;
and the tapes differ (else this compares a task with itself).
"""

from std.testing import assert_equal, assert_true

from noeira.physics3d.gpu.constants import META_IDX_TASK_PARAM_0
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.posed_reset import task_meta_words
from noeira.tasks.tape import TAPE_WORDS

comptime CFG = So101TowerConfig


def _words(task: String) raises -> Dict[Int, Float64]:
    var mw = task_meta_words(
        task, String("so101_tower"), CFG.SHAPE_W_GOAL, CFG.SHAPE_W_REACH,
        CFG.GOAL_MARGIN, CFG.REACH_MARGIN,
    )
    var d = Dict[Int, Float64]()
    var idx = mw[0].copy()
    var val = mw[1].copy()
    for k in range(len(idx)):
        d[idx[k]] = Float64(val[k])
    return d^


def main() raises:
    var a = _words(String("so101_tower_cube_in_bowl"))
    var b = _words(String("so101_tower_lift_real_layout"))
    assert_equal(len(a), len(b), "the same meta words are written")
    var same = 0
    var tape_diff = 0
    for e in a.items():
        var k = e.key
        assert_true(k in b, "word " + String(k) + " written by both")
        var in_tape = k >= META_IDX_TASK_PARAM_0 and k < META_IDX_TASK_PARAM_0 + TAPE_WORDS
        if in_tape:
            if e.value != b[k]:
                tape_diff += 1
        else:
            assert_equal(e.value, b[k], "word " + String(k) + " (not the tape)")
            same += 1
    assert_true(tape_diff > 0, "the goal tapes differ")
    print("  ", same, "non-tape words equal,", tape_diff, "tape words differ")
    print("REAL LAYOUT TASK OK")
