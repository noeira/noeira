"""A tracker trained with the reference skip is refused by the readers that
would drop it.

`SuperMimicTrainer.act` and the tracker's imagined rollout add the reference
skip (`lanes.sm_ref_skip`). The prompt interface and the object encoder run
the tracker's decoder themselves and do not: handed such a tracker they would
command the decoder's correction alone, with no error. They raise instead
(`SuperMimicTrainer.require_no_ref_skip`).

[1] the prompt trainer: `act` and `policy_update` run on a tracker without
    the skip (the refusal is not unconditional), raise once the tracker has
    it, and run again once it is taken off.
[2] the object trainer: the same three readings.
[3] the reset rule both envs share (`SMResetSampler.reset_frame`), on the CPU
    env: a `restart`, whose lanes all take a queued frame, leaves the host's
    random stream where it was; a reset that draws its frame moves it (the
    check can fail).

Run: pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_skip_guard.mojo
"""

from std.random import random_ui64, seed

from noeira.deep_agents.supermimic.config import SMConfig
from noeira.deep_agents.supermimic.env import SMTrackingEnv
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_imagination import SMObjImagBatch
from noeira.deep_agents.supermimic.object_policy import SMObjConfig
from noeira.deep_agents.supermimic.object_spec import SM_OBJ
from noeira.deep_agents.supermimic.object_task import SM_OREF
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer
from noeira.deep_agents.supermimic.prompt_imagination import SMPromptBatch
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfig
from noeira.deep_agents.supermimic.prompt_trainer import SMPromptTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, SMDims
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime R = G1MjlabRobot
comptime D = SMDims[R]
comptime N = 4
comptime H = 4
comptime NREF = 2
comptime CMD = 7
comptime C = SMConfig["cpu", R, 32, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, N, 8, N, H]
comptime P = SMPromptConfig[C, CMD, 32, 32, 32, 16]
comptime OC = SMObjConfig[C, 32, 32, 32, 32, 16]


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _prompt_act(mut tr: SMPromptTrainer[P]) -> Bool:
    """True when `act` ran."""
    var hist = TensorImpl[DT].alloc(N * D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(N * D.GOAL)
    var cmds = TensorImpl[DT].alloc(N * CMD)
    try:
        _ = tr.act[N](hist, goals, cmds, 2)
        return True
    except e:
        return False


def _prompt_update(mut tr: SMPromptTrainer[P]) -> Bool:
    var ib = SMPromptBatch[R, CMD](N, H, NREF)
    for b in range(N):
        ib.state.data[b * D.S + 2] = 0.76
        ib.state.data[b * D.S + 3] = 1
    var none = List[Int]()
    var vals = List[Float64]()
    var tgts = List[Float64]()
    try:
        _ = tr.policy_update[N](ib, none, vals, tgts)
        return True
    except e:
        return False


def _obj_act(mut tr: SMObjTrainer[OC]) -> Bool:
    var hist = TensorImpl[DT].alloc(N * D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(N * D.GOAL)
    var s = TensorImpl[DT].alloc(N * D.S)
    var o = TensorImpl[DT].alloc(N * SM_OBJ)
    var oref = TensorImpl[DT].alloc(N * SM_OREF)
    for b in range(N):
        s.data[b * D.S + 2] = 0.76
        s.data[b * D.S + 3] = 1
        o.data[b * SM_OBJ + 3] = 1
    try:
        _ = tr.act[N](hist, goals, s, o, oref)
        return True
    except e:
        return False


def _obj_update(mut tr: SMObjTrainer[OC]) -> Bool:
    var ib = SMObjImagBatch[R](N, H)
    for b in range(N):
        ib.base.state.data[b * D.S + 2] = 0.76
        ib.base.state.data[b * D.S + 3] = 1
        ib.obj0.data[b * SM_OBJ + 3] = 1
    var none = List[Int]()
    var vals = List[Float64]()
    var tgts = List[Float64]()
    try:
        _ = tr.policy_update[N](ib, none, vals, tgts)
        return True
    except e:
        return False


def _library() raises -> SMMotionLibrary[R]:
    """Two clips of the default pose standing at 0.76 m."""
    var lens: List[Int] = [40, 30]
    var rows = List[Scalar[DT]](length=70 * D.S, fill=0)
    for f in range(70):
        rows[f * D.S + 2] = 0.76
        rows[f * D.S + 3] = 1
        for j in range(R.NDOF):
            rows[f * D.S + 7 + j] = Scalar[DT](R.default_joint_pos(j))
    var q0 = List[Float64](length=R.NDOF, fill=0.0)
    for j in range(R.NDOF):
        q0[j] = R.default_joint_pos(j)
    return SMMotionLibrary[R](rows^, lens, q0)


def main() raises:
    var t = Tally()

    print("[1] the prompt trainer")
    var pt = SMPromptTrainer[P](SuperMimicTrainer[C].for_robot(R.SIM.create()))
    t.truth(_prompt_act(pt), "act runs on a tracker without the skip")
    t.truth(_prompt_update(pt), "policy_update runs on it")
    pt.base.set_ref_skip(True)
    t.truth(not _prompt_act(pt), "act refuses a tracker with the skip")
    t.truth(not _prompt_update(pt), "policy_update refuses it")
    pt.base.set_ref_skip(False)
    t.truth(_prompt_act(pt), "act runs again with the skip off")

    print("\n[2] the object trainer")
    var ot = SMObjTrainer[OC](SuperMimicTrainer[C].for_robot(R.SIM.create()))
    t.truth(_obj_act(ot), "act runs on a tracker without the skip")
    t.truth(_obj_update(ot), "policy_update runs on it")
    ot.base.set_ref_skip(True)
    t.truth(not _obj_act(ot), "act refuses a tracker with the skip")
    t.truth(not _obj_update(ot), "policy_update refuses it")
    ot.base.set_ref_skip(False)
    t.truth(_obj_act(ot), "act runs again with the skip off")

    print("\n[3] a queued reset draws nothing from the host's random stream")
    var env = SMTrackingEnv[R](N, _library(), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0)
    seed(91)
    var want = random_ui64(0, 0xFFFFFFFF)
    var frames = List[Int](length=N, fill=0)
    for l in range(N):
        frames[l] = 40 if l % 2 == 1 else 0
    seed(91)
    env.restart(frames)
    var got = random_ui64(0, 0xFFFFFFFF)
    t.truth(got == want, "after a restart on queued frames the stream's next draw is unchanged")
    var on_frames = True
    for l in range(N):
        on_frames = on_frames and env.lanes[l].frame == frames[l] + 1
    t.truth(on_frames, "and every lane is on its queued frame (+1, the command update)")
    for l in range(N):
        env.next_frame[l] = -1
    var all = List[Int]()
    for l in range(N):
        all.append(l)
    seed(91)
    env._reset(all)
    var moved = random_ui64(0, 0xFFFFFFFF)
    t.truth(moved != want, "a reset that draws its frames moves the stream (the check above can fail)")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_skip_guard: " + String(t.fails) + " failed")
