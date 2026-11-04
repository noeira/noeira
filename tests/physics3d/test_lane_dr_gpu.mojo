"""Per-lane domain randomisation (`Data.lane_dr`, `LANE_DR=True`) against the
same values written into the model, on the batched dm_control quadruped (a
free-jointed torso, four feet in contact).

ONE batch of 2 lanes with LANE_DR: lane l gets friction μ_l on every body's
contacts and a CoM offset (0, 0, δ_l) on the torso. Against it, for each l,
a plain 2-lane batch (LANE_DR=False) whose MODEL carries μ_l on every geom
(MuJoCo mixes equal priorities by max, so every contact's friction is μ_l)
and the torso's body_ipos + δ_l, read at the SAME lane l. Same start state,
same actions, 40 control steps.

⚠ LANE-MATCHED ON PURPOSE. In the plain batched quadruped, lane 1 departs
from lane 0 at the second step from an identical state and actions (lane 0
equals a 1-lane batch bit for bit; measured 2026-11-04, before any LANE_DR
code ran): something in the step is lane-dependent. Comparing lane l with
lane l keeps that out of this gate; it is its own question.

  [1] each lane's trajectory (qpos, qvel) equals its model-edited twin's
      BIT FOR BIT: the override is the model value, applied per lane
  [2] the overrides act: the two lanes' trajectories differ, and each
      differs from the unedited model's
  [3] zero is the model: a LANE_DR batch with `lane_dr` all zero equals
      the unedited plain batch, lane by lane, bit for bit

Run: pixi run -e apple mojo run -I . tests/physics3d/test_lane_dr_gpu.mojo  (or -e nvidia)
"""

from std.random import random_float64, seed

from max.gpu.host import DeviceContext

from noeira.envs.dm_control.quadruped.quadruped import DMQuadrupedWalkConfig, DMQuadrupedWalkModel
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.nn.constants import DT
from noeira.physics3d.constraints.lane_dr import LANE_DR_FRICTION, LANE_DR_IPOS_X, LANE_DR_STRIDE
from noeira.physics3d.gpu.constants import BODY_IDX_IPOS_X, GEOM_IDX_FRICTION, MODEL_BODY_SIZE, MODEL_GEOM_SIZE


comptime QEnv[NE: Int, LD: Bool] = Phyics3dBatchedEnv[
    DMQuadrupedWalkModel, DMQuadrupedWalkConfig, NE, TERMINATE_ON_UNHEALTHY=False, LANE_DR=LD,
]
comptime NQ = QEnv[1, False].NQ
comptime NV = QEnv[1, False].NV
comptime NA = QEnv[1, False].ACT_DIM
comptime TORSO = 1
comptime STEPS = 40


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


def _start[NE: Int, LD: Bool](mut env: QEnv[NE, LD], ctx: DeviceContext, q0: List[Scalar[DT]], v0: List[Scalar[DT]]) raises:
    """Every lane at (q0, v0), kinematics refreshed."""
    env.d.qpos.download(ctx)
    env.d.qvel.download(ctx)
    for l in range(NE):
        for k in range(NQ):
            env.d.qpos.data[l * NQ + k] = q0[k]
        for k in range(NV):
            env.d.qvel.data[l * NV + k] = v0[k]
    env.d.qpos.upload_resident(ctx)
    env.d.qvel.upload_resident(ctx)
    env._run_fields_fk(ctx)
    env._run_fields_vel(ctx)


def _roll[NE: Int, LD: Bool](mut env: QEnv[NE, LD], ctx: DeviceContext, acts: List[List[Scalar[DT]]]) raises -> List[List[Scalar[DT]]]:
    """STEPS steps, every lane the same actions; per lane the concatenated
    (qpos, qvel) after each step."""
    var out = List[List[Scalar[DT]]]()
    for _ in range(NE):
        out.append(List[Scalar[DT]]())
    var ha = ctx.enqueue_create_host_buffer[DT](NE * NA)
    ctx.synchronize()
    for t in range(STEPS):
        for l in range(NE):
            for j in range(NA):
                ha[l * NA + j] = acts[t][j]
        ctx.enqueue_copy(env._action, ha)
        env.step_batch[NE](Optional(ctx), UInt64(t))
        env.d.qpos.download(ctx)
        env.d.qvel.download(ctx)
        for l in range(NE):
            for k in range(NQ):
                out[l].append(env.d.qpos.data[l * NQ + k])
            for k in range(NV):
                out[l].append(env.d.qvel.data[l * NV + k])
    return out^


def _set_lane_dr(mut env: QEnv[2, True], ctx: DeviceContext, mu: List[Float64], dz: List[Float64]) raises:
    """Lane l: friction mu[l] on every body (≤ 0: none), torso CoM + (0, 0, dz[l])."""
    var nb = env.mf.bodies.n // MODEL_BODY_SIZE
    env.d.lane_dr.download(ctx)
    for i in range(env.d.lane_dr.n):
        env.d.lane_dr.data[i] = 0
    for l in range(2):
        for b in range(nb):
            env.d.lane_dr.data[(l * nb + b) * LANE_DR_STRIDE + LANE_DR_FRICTION] = Scalar[DT](mu[l])
        env.d.lane_dr.data[(l * nb + TORSO) * LANE_DR_STRIDE + LANE_DR_IPOS_X + 2] = Scalar[DT](dz[l])
    env.d.lane_dr.upload_resident(ctx)


def _edit_model(mut env: QEnv[2, False], ctx: DeviceContext, mu: Float64, dz: Float64) raises:
    """The same values in the shared model: every geom's friction, the torso's ipos."""
    if mu > 0:
        var ng = env.mf.geoms.n // MODEL_GEOM_SIZE
        for g in range(ng):
            env.mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_FRICTION] = Scalar[DT](mu)
        env.mf.geoms.upload_resident(ctx)
    var o = TORSO * MODEL_BODY_SIZE + BODY_IDX_IPOS_X + 2
    env.mf.bodies.data[o] = env.mf.bodies.data[o] + Scalar[DT](dz)
    env.mf.bodies.upload_resident(ctx)


def _diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Tuple[Int, Float64]:
    """(mismatching entries, max |a − b|)."""
    var n = 0
    var m = 0.0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
            m = max(m, abs(Float64(a[i]) - Float64(b[i])))
    return (n, m)


def main() raises:
    var ctx = DeviceContext()
    var tl = Tally()
    seed(3)
    var acts = List[List[Scalar[DT]]]()
    for _ in range(STEPS):
        var a = List[Scalar[DT]](length=NA, fill=0)
        for j in range(NA):
            a[j] = Scalar[DT](random_float64(-1, 1))
        acts.append(a^)
    # the start: the plain env's reset lane 0
    var e0 = QEnv[1, False](ctx)
    e0.reset_batch[1](Optional(ctx), 0)
    e0.d.qpos.download(ctx)
    e0.d.qvel.download(ctx)
    var q0 = List[Scalar[DT]](length=NQ, fill=0)
    var v0 = List[Scalar[DT]](length=NV, fill=0)
    for k in range(NQ):
        q0[k] = e0.d.qpos.data[k]
    for k in range(NV):
        v0[k] = e0.d.qvel.data[k]
    var mu: List[Float64] = [0.35, 1.4]
    var dz: List[Float64] = [0.03, -0.04]

    print("[1] lane overrides vs the values in the model")
    var edr = QEnv[2, True](ctx)
    edr.reset_batch[2](Optional(ctx), 0)
    _set_lane_dr(edr, ctx, mu, dz)
    _start[2, True](edr, ctx, q0, v0)
    var tdr = _roll[2, True](edr, ctx, acts)
    var tm = List[List[Scalar[DT]]]()
    for l in range(2):
        var em = QEnv[2, False](ctx)
        em.reset_batch[2](Optional(ctx), 0)
        _edit_model(em, ctx, mu[l], dz[l])
        _start[2, False](em, ctx, q0, v0)
        var r = _roll[2, False](em, ctx, acts)
        tm.append(r[l].copy())
    for l in range(2):
        var d = _diff(tdr[l], tm[l])
        tl.check(d[0] == 0, "lane " + String(l) + " (mu " + String(mu[l]) + ", dz " + String(dz[l]) + "): " + String(len(tdr[l])) + " values over " + String(STEPS) + " steps, " + String(d[0]) + " differ from the model-edited twin (max " + String(d[1]) + ")")

    print("[2] the overrides act")
    var eb = QEnv[2, False](ctx)
    eb.reset_batch[2](Optional(ctx), 0)
    _start[2, False](eb, ctx, q0, v0)
    var tb = _roll[2, False](eb, ctx, acts)
    var d01 = _diff(tdr[0], tdr[1])
    var d0b = _diff(tdr[0], tb[0])
    var d1b = _diff(tdr[1], tb[1])
    tl.check(d01[1] > 1e-3 and d0b[1] > 1e-3 and d1b[1] > 1e-3, "lane 0 vs lane 1: max " + String(d01[1]) + "; vs the unedited model: " + String(d0b[1]) + ", " + String(d1b[1]))

    print("[3] zero is the model")
    var ez = QEnv[2, True](ctx)
    ez.reset_batch[2](Optional(ctx), 0)
    var zero: List[Float64] = [0.0, 0.0]
    _set_lane_dr(ez, ctx, zero, zero)
    _start[2, True](ez, ctx, q0, v0)
    var tz = _roll[2, True](ez, ctx, acts)
    var dz0 = _diff(tz[0], tb[0])
    var dz1 = _diff(tz[1], tb[1])
    tl.check(dz0[0] == 0 and dz1[0] == 0, "LANE_DR with zeros: " + String(dz0[0] + dz1[0]) + " values differ from the plain batch")

    print("\n===", tl.ok, "/", tl.total, "passed ===")
    if tl.ok != tl.total:
        raise Error("test_lane_dr_gpu: " + String(tl.total - tl.ok) + " failed")
