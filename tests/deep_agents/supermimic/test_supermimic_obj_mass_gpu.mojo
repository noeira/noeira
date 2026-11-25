"""The object's mass per lane in the device env that carries it
(`object_env_gpu.SMObjTrackingEnvGpu`, its `mass_lo` / `mass_hi`): the draw
at a reset, what it writes, and that it lasts until the lane's next reset.

NVIDIA ONLY (the batched G1 does not build for Metal). The PHYSICS of a
per-lane mass is gated elsewhere (`tests/physics3d/test_lane_mass*.mojo`, the
batched G1 with its crate among them); this file gates the WIRING. Built with
`-D SM_LANE_MASS=1`:

  [1] a range of (0.5, 3): after the first reset, every lane's context mass
      is the model's times a scale in the range, the lanes are not alike
      (64 lanes: the smallest under 1, the largest over 2.5, the mean within
      1.75 +- 0.4), the simulator's `lane_mass` holds that scale at the box
      and 0 at every other body, and its per-lane solver weights are the
      model's, the box's divided by its scale
  [2] every lane restarted well inside its clip, then 5 steps: no lane
      ends and no scale changes; after the steps, a restart draws again.
      (As first written this check stepped from the env's own first reset,
      which starts lanes anywhere in their clips: 3 of 64 were within 5
      frames of their clip's end, ended, and drew again, as they must. The
      check called that a failure; it was its own premise that was wrong.)
  [3] the same value twice (2, 2): 2 in every lane, exactly
  [4] (1, 1), the default: nothing is written (`lane_mass` all zero), the
      context is the model's mass

Built WITHOUT the define: a mass range is refused, and (1, 1) builds as before.

Run (box): pixi run mojo build -D SM_LANE_MASS=1 -I . -o build/t tests/deep_agents/supermimic/test_supermimic_obj_mass_gpu.mojo && pixi run ./build/t
"""

from std.sys import default_accelerator

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.env_gpu import SMReplayGpuOf
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_env_gpu import SMObjReplayGpu, SMObjTrackingEnvGpu, sm_object_track
from noeira.deep_agents.supermimic.object_spec import SM_OBJ_CTX
from noeira.deep_agents.supermimic.robot import G1MjlabBoxBatchedSim, G1MjlabBoxRobot, SM_LANE_MASS, SMDims
from noeira.envs.robots.unitree_g1_mjlab_box_xml import G1M_BOX_REST_Z
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime R = G1MjlabBoxRobot
comptime D = SMDims[R]
comptime n = R.NDOF
comptime NL = 64
comptime BS = G1MjlabBoxBatchedSim[NL]
comptime EnvO = SMObjTrackingEnvGpu[R, BS]
comptime NB = BS.SIM_NBODY
comptime FPS = 50.0
comptime FAR_X = 3.0          # the box rests out of the robot's reach (`test_supermimic_obj_env_gpu`)
comptime SETTLE_STEPS = 100


struct Tally:
    var ok: Int
    var total: Int

    def __init__(out self):
        self.ok = 0
        self.total = 0

    def check(mut self, cond: Bool, msg: String):
        self.total += 1
        if cond:
            self.ok += 1
            print("  ok:", msg)
        else:
            print("  FAIL:", msg)


def _standing_row() raises -> List[Scalar[DT]]:
    var sim = R.SIM.create()
    var far: List[Float64] = [FAR_X, 0.0, G1M_BOX_REST_Z, 1.0, 0.0, 0.0, 0.0]
    sim.set_box(far, List[Float64](length=6, fill=0.0))
    var a = List[Float64](length=n, fill=0.0)
    for _ in range(SETTLE_STEPS):
        sim.control_step(a, a)
    var row = List[Scalar[DT]](length=D.S, fill=0)
    for k in range(R.NQ):
        row[k] = Scalar[DT](sim.qpos(k))
    return row^


def _library(stand: List[Scalar[DT]], lens: List[Int]) raises -> SMMotionLibrary[R]:
    var total = 0
    for c in range(len(lens)):
        total += lens[c]
    var rows = List[Scalar[DT]](length=total * D.S, fill=0)
    for f in range(total):
        for k in range(D.S):
            rows[f * D.S + k] = stand[k]
    var q0 = List[Float64](length=n, fill=0.0)
    for j in range(n):
        q0[j] = R.default_joint_pos(j)
    return SMMotionLibrary[R](rows^, lens, q0)


def _resting_poses(lens: List[Int]) -> List[Float64]:
    var out = List[Float64]()
    for c in range(len(lens)):
        for _ in range(lens[c]):
            out.append(FAR_X)
            out.append(0.0)
            out.append(G1M_BOX_REST_Z)
            out.append(1.0)
            out.append(0.0)
            out.append(0.0)
            out.append(0.0)
    return out^


def _env(ctx: DeviceContext, stand: List[Scalar[DT]], lo: Float64, hi: Float64) raises -> EnvO:
    """Long clips (no lane ends within the test), no noise."""
    var lens: List[Int] = [400, 400]
    var track = sm_object_track(_resting_poses(lens), lens, FPS)
    return EnvO(
        ctx, _library(stand, lens), track, BS.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0,
        friction_dr=False, max_episode_steps=0, mass_lo=lo, mass_hi=hi,
    )


def _ctx_mass(ctx: DeviceContext, mut env: EnvO) raises -> List[Float64]:
    env.octx.download(ctx)
    var out = List[Float64]()
    for l in range(NL):
        out.append(Float64(env.octx.data[l * SM_OBJ_CTX]))
    return out^


def _scales(ctx: DeviceContext, mut env: EnvO) raises -> List[Float64]:
    """The simulator's `lane_mass`, [NL · NB]."""
    env.env.sim.env.d.lane_mass.download(ctx)
    var out = List[Float64]()
    for i in range(NL * NB):
        out.append(Float64(env.env.sim.env.d.lane_mass.data[i]))
    return out^


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print("SKIP test_supermimic_obj_mass_gpu: needs an NVIDIA accelerator (the batched G1)")
        return
    else:
        var ctx = DeviceContext()
        var tl = Tally()
        var stand = _standing_row()
        print("built with SM_LANE_MASS:", SM_LANE_MASS, " lanes", NL, " bodies", NB, " the box is body", EnvO.OB)
        comptime if not SM_LANE_MASS:
            var refused = False
            try:
                var e = _env(ctx, stand, 0.5, 3.0)
            except err:
                refused = True
            tl.check(refused, "without -D SM_LANE_MASS=1 a mass range is refused")
            var plain = _env(ctx, stand, 1.0, 1.0)
            var cm = _ctx_mass(ctx, plain)
            var off = 0.0
            for l in range(NL):
                off = max(off, abs(cm[l] - plain.mass))
            tl.check(off == 0.0, "(1, 1) builds as before: the context is the model's mass in every lane")
        else:
            print("[1] a range of (0.5, 3)")
            var env = _env(ctx, stand, 0.5, 3.0)
            var m0 = env.mass
            var cm = _ctx_mass(ctx, env)
            var sc = _scales(ctx, env)
            var lo = 1e9
            var hi = 0.0
            var mean = 0.0
            var eb = 0.0
            var stray = 0
            for l in range(NL):
                var s = cm[l] / m0
                lo = min(lo, s)
                hi = max(hi, s)
                mean += s / Float64(NL)
                eb = max(eb, abs(sc[l * NB + EnvO.OB] - s))
                for b in range(NB):
                    if b != EnvO.OB and sc[l * NB + b] != 0.0:
                        stray += 1
            tl.check(lo >= 0.5 and hi <= 3.0, "every lane's scale is in the range: " + String(lo) + " to " + String(hi))
            tl.check(lo < 1.0 and hi > 2.5 and abs(mean - 1.75) < 0.4, "the lanes are spread: smallest " + String(lo) + ", largest " + String(hi) + ", mean " + String(mean))
            tl.check(eb <= 1e-6 and stray == 0, "lane_mass holds the context's scale at the box (max |d| " + String(eb) + ") and nothing elsewhere (" + String(stray) + " other entries set)")
            env.env.sim.env.d.lane_body_invweight0.download(ctx)
            var ew = 0.0
            var eo = 0.0
            for l in range(NL):
                for b in range(NB):
                    for k in range(2):
                        var got = Float64(env.env.sim.env.d.lane_body_invweight0.data[(l * NB + b) * 2 + k])
                        var mod = Float64(env.env.sim.env.mf.body_invweight0.data[b * 2 + k])
                        if b == EnvO.OB:
                            var want = mod / sc[l * NB + b]
                            ew = max(ew, abs(got - want) / want)
                        else:
                            eo = max(eo, abs(got - mod))
            tl.check(ew <= 1e-6 and eo == 0.0, "the lanes' solver weights: the box's is the model's over its scale (rel. " + String(ew) + "), the others the model's (max |d| " + String(eo) + ")")

            print("[2] a scale lasts until the lane's next reset")
            var rb = SMReplayGpuOf[R](ctx, NL, 16)
            var orb = SMObjReplayGpu(ctx, NL, 16)
            var act = TensorImpl[DT].alloc(NL * n)
            act.upload_resident(ctx)
            var frames = List[Int](length=NL, fill=10)
            env.restart(ctx, frames)
            sc = _scales(ctx, env)
            var ended = 0
            for _ in range(5):
                var si = env.step(ctx, act, rb, orb)
                ended += si.n_terminated + si.n_truncated
            var sc5 = _scales(ctx, env)
            var moved = 0
            var set5 = 0
            for i in range(NL * NB):
                if sc5[i] != sc[i]:
                    moved += 1
                if sc5[i] != 0.0:
                    set5 += 1
            tl.check(ended == 0 and moved == 0 and set5 == NL, "every lane at frame 10 of 400, 5 steps: " + String(ended) + " lanes ended, " + String(moved) + " scales changed, " + String(set5) + " lanes hold one")
            env.restart(ctx, frames)
            var sc_r = _scales(ctx, env)
            var cm_r = _ctx_mass(ctx, env)
            var redrawn = 0
            var er = 0.0
            for l in range(NL):
                if sc_r[l * NB + EnvO.OB] != sc[l * NB + EnvO.OB]:
                    redrawn += 1
                er = max(er, abs(cm_r[l] / m0 - sc_r[l * NB + EnvO.OB]))
            tl.check(redrawn >= NL - 2 and er <= 1e-6, "after the steps a restart draws again: " + String(redrawn) + " of " + String(NL) + " lanes changed, context and scale agree (max |d| " + String(er) + ")")

            print("[3] the same value twice")
            var e2 = _env(ctx, stand, 2.0, 2.0)
            var cm2 = _ctx_mass(ctx, e2)
            var sc2 = _scales(ctx, e2)
            var bad = 0
            for l in range(NL):
                if cm2[l] != 2.0 * m0 or sc2[l * NB + EnvO.OB] != 2.0:
                    bad += 1
            tl.check(bad == 0, "(2, 2): a scale of 2 and a context of twice the model's mass in every lane (" + String(bad) + " lanes off)")

            print("[4] (1, 1), the default")
            var e1 = _env(ctx, stand, 1.0, 1.0)
            var cm1 = _ctx_mass(ctx, e1)
            var sc1 = _scales(ctx, e1)
            var set1 = 0
            var off1 = 0.0
            for i in range(NL * NB):
                if sc1[i] != 0.0:
                    set1 += 1
            for l in range(NL):
                off1 = max(off1, abs(cm1[l] - m0))
            tl.check(set1 == 0 and off1 == 0.0, "nothing written (" + String(set1) + " scales set), the context is the model's mass")

        print("\n===", tl.ok, "/", tl.total, "passed ===")
        if tl.ok != tl.total:
            raise Error("test_supermimic_obj_mass_gpu: " + String(tl.total - tl.ok) + " failed")
