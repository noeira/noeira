"""A per-lane mass for the crate of the batched G1 (`Phyics3dBatchedEnv[...,
LANE_MASS=True]` on mjlab's G1 with its free box, the env SuperMimic's object
network trains in), 2 lanes, float32.

What a per-lane mass is, and its gates on a small fixture:
`noeira/physics3d/dynamics/lane_mass.mojo`, `tests/physics3d/test_lane_mass.mojo`
(CPU, float64), `test_lane_mass_gpu.mojo` (device). This file asks the same of
the real env, where NVIDIA runs the BLOCKED Newton kernel.

The crate is dropped on the robot's head from 10 cm, so that its mass acts
through contacts with the robot within the run (a crate at rest on the floor
moves alike whatever it weighs).

  [1] `step_batch`, 25 control steps at a zero action, scales that are powers
      of two (lane 0: x2, lane 1: x4): each lane equals its TWIN, a plain env
      (`LANE_MASS=False`) whose model holds that mass (mass, inertia, their
      inverses, the solver's weight), BIT FOR BIT; and is not the unscaled
      model's
  [2] no scale: the plain env, bit for bit
  [3] the solver's weights are live: a twin without them differs
  [4] the scales written alone on the device buffer, then `sync_lane_mass`:
      the host setter's run, bit for bit
  [5] the env's integrator on the device against the same type on the CPU,
      60 raw steps with a scale of 3 in lane 1: e <= max(10 e0, 1e-5), e0 the
      distance with no scale

NVIDIA ONLY: the batched G1 does not build for Metal (its action hook
computes in float64), so this prints SKIP elsewhere.

Run: pixi run -e nvidia mojo run -I . tests/physics3d/test_lane_mass_g1_gpu.mojo
"""

from std.math import abs
from std.sys import default_accelerator

from max.gpu.host import DeviceContext

from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.robots.unitree_g1_mjlab_box_xml import G1M_BOX_QPOS, G1M_BOX_QVEL, UnitreeG1MjlabBoxModel
from noeira.envs.robots.unitree_g1_mjlab_config import UnitreeG1MjlabConfig
from noeira.nn.constants import DT
from noeira.physics3d.dynamics.lane_mass import lane_mass_reset, set_lane_mass_scale
from noeira.physics3d.fields import Data
from noeira.physics3d.gpu.constants import BODY_IDX_INV_IXX, BODY_IDX_INV_MASS, BODY_IDX_IXX, BODY_IDX_MASS, MODEL_BODY_SIZE


comptime NL = 2
comptime GEnv[LM: Bool] = Phyics3dBatchedEnv[
    UnitreeG1MjlabBoxModel, UnitreeG1MjlabConfig, NL, TERMINATE_ON_UNHEALTHY=False, LANE_DR=True, LANE_MASS=LM,
]
comptime GData = Data[DT, GEnv[True].MD, NL]
comptime NQ = GEnv[True].NQ
comptime NV = GEnv[True].NV
comptime NA = GEnv[True].ACT_DIM
comptime NB = GEnv[True].NBODY
comptime BOX = NB - 1                 # the box is the model's last body
comptime ROW = NQ + NV
comptime ENV_STEPS = 25
comptime RAW_STEPS = 60
comptime W_SETTER = 0
comptime W_KERNEL = 1


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


def _state(base: List[Scalar[DT]], mut d: GData):
    """Both lanes at the reset pose, at rest, the crate 10 cm over the head
    (lane 1's a little to the side)."""
    for l in range(NL):
        for k in range(NQ):
            d.qpos.data[l * NQ + k] = base[k]
        for k in range(NV):
            d.qvel.data[l * NV + k] = Scalar[DT](0)
            d.qacc.data[l * NV + k] = Scalar[DT](0)
            d.qacc_warmstart.data[l * NV + k] = Scalar[DT](0)
            d.qfrc.data[l * NV + k] = Scalar[DT](0)
        var o = l * NQ + G1M_BOX_QPOS
        d.qpos.data[o + 0] = base[0] + Scalar[DT](0.02)
        d.qpos.data[o + 1] = base[1] + Scalar[DT](0.03 * Float64(l))
        d.qpos.data[o + 2] = base[2] + Scalar[DT](0.84)
        d.qpos.data[o + 3] = Scalar[DT](1)
        d.qpos.data[o + 4] = Scalar[DT](0)
        d.qpos.data[o + 5] = Scalar[DT](0)
        d.qpos.data[o + 6] = Scalar[DT](0)


def _edit_model[LM: Bool](mut env: GEnv[LM], ctx: DeviceContext, s: Float64, weights: Bool) raises:
    """The env's shared model holds the box's mass times `s`."""
    var o = BOX * MODEL_BODY_SIZE
    var sc = Scalar[DT](s)
    env.mf.bodies.data[o + BODY_IDX_MASS] = env.mf.bodies.data[o + BODY_IDX_MASS] * sc
    env.mf.bodies.data[o + BODY_IDX_INV_MASS] = env.mf.bodies.data[o + BODY_IDX_INV_MASS] / sc
    for k in range(3):
        env.mf.bodies.data[o + BODY_IDX_IXX + k] = env.mf.bodies.data[o + BODY_IDX_IXX + k] * sc
        env.mf.bodies.data[o + BODY_IDX_INV_IXX + k] = env.mf.bodies.data[o + BODY_IDX_INV_IXX + k] / sc
    env.mf.bodies.upload_resident(ctx)
    if weights:
        for k in range(2):
            env.mf.body_invweight0.data[BOX * 2 + k] = env.mf.body_invweight0.data[BOX * 2 + k] / sc
        env.mf.body_invweight0.upload_resident(ctx)


def _rows(d: GData, mut out: List[Scalar[DT]]):
    for l in range(NL):
        for k in range(NQ):
            out.append(d.qpos.data[l * NQ + k])
        for k in range(NV):
            out.append(d.qvel.data[l * NV + k])


def _start[LM: Bool](mut env: GEnv[LM], ctx: DeviceContext, base: List[Scalar[DT]], scales: List[Float64], how: Int) raises:
    """A fresh episode, the state on the device, and (LM) lane l's scale
    `scales[l]` (0 = none) by the host setter or by the device kernel."""
    env.reset_batch[NL](Optional(ctx), 0)
    _state(base, env.d)
    env.d.qpos.upload_resident(ctx)
    env.d.qvel.upload_resident(ctx)
    env.d.qacc.upload_resident(ctx)
    env.d.qacc_warmstart.upload_resident(ctx)
    env.d.qfrc.upload_resident(ctx)
    comptime if LM:
        lane_mass_reset(env.d, env.mf)
        for l in range(len(scales)):
            if scales[l] > 0.0:
                if how == W_SETTER:
                    set_lane_mass_scale(env.d, env.mf, l, BOX, scales[l])
                else:
                    env.d.lane_mass.data[l * NB + BOX] = Scalar[DT](scales[l])
        env.d.lane_mass.upload_resident(ctx)
        env.d.lane_body_invweight0.upload_resident(ctx)
        if how == W_KERNEL:
            _ = env.lane_mass_ptr()
            env.sync_lane_mass(ctx)
    env._run_fields_fk(ctx)
    env._run_fields_vel(ctx)


def _env_run[LM: Bool](mut env: GEnv[LM], ctx: DeviceContext, base: List[Scalar[DT]], scales: List[Float64], how: Int) raises -> List[Scalar[DT]]:
    """`step_batch` at a zero action, ENV_STEPS control steps."""
    _start[LM](env, ctx, base, scales, how)
    var ha = ctx.enqueue_create_host_buffer[DT](NL * NA)
    ctx.synchronize()
    for i in range(NL * NA):
        ha[i] = Scalar[DT](0)
    for t in range(ENV_STEPS):
        ctx.enqueue_copy(env._action, ha)
        env.step_batch[NL](Optional(ctx), UInt64(t))
    env.d.qpos.download(ctx)
    env.d.qvel.download(ctx)
    var out = List[Scalar[DT]]()
    _rows(env.d, out)
    return out^


def _raw_gpu(mut env: GEnv[True], ctx: DeviceContext, base: List[Scalar[DT]], scales: List[Float64]) raises -> List[Scalar[DT]]:
    """The env's integrator alone on the device, RAW_STEPS steps, no actuation."""
    _start[True](env, ctx, base, scales, W_SETTER)
    for _ in range(RAW_STEPS):
        env.integ_euler.step["gpu"](env.d, env.mf, ctx)
    env.d.qpos.download(ctx)
    env.d.qvel.download(ctx)
    var out = List[Scalar[DT]]()
    _rows(env.d, out)
    return out^


def _raw_cpu(mut env: GEnv[True], base: List[Scalar[DT]], scales: List[Float64]) raises -> List[Scalar[DT]]:
    """The same integrator type on the CPU, on the env's model."""
    var d = GData()
    _state(base, d)
    lane_mass_reset(d, env.mf)
    for l in range(len(scales)):
        if scales[l] > 0.0:
            set_lane_mass_scale(d, env.mf, l, BOX, scales[l])
    var integ = GEnv[True].IntegEuler()
    for _ in range(RAW_STEPS):
        integ.step["cpu"](d, env.mf)
    var out = List[Scalar[DT]]()
    _rows(d, out)
    return out^


def _dist(a: List[Scalar[DT]], b: List[Scalar[DT]], lo: Int, hi: Int) -> Tuple[Int, Float64]:
    var n = 0
    var w = 0.0
    for i in range(lo, hi):
        if a[i] != b[i]:
            n += 1
            w = max(w, abs(Float64(a[i]) - Float64(b[i])))
    return (n, w)


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        # the batched G1 is NVIDIA only; never instantiated here
        print("SKIP test_lane_mass_g1_gpu: needs an NVIDIA accelerator (the batched G1)")
        return
    else:
        var ctx = DeviceContext()
        var tl = Tally()
        var env = GEnv[True](ctx)
        env.reset_batch[NL](Optional(ctx), 0)
        env.d.qpos.download(ctx)
        var base = List[Scalar[DT]](length=NQ, fill=0)
        for k in range(NQ):
            base[k] = env.d.qpos.data[k]
        print("root height", base[2], " bodies", NB, " the box is body", BOX, " nq", NQ, " nv", NV)

        print("[1] lanes against twin envs (the mass in the model)")
        var sc: List[Float64] = [2.0, 4.0]
        var none: List[Float64] = [0.0, 0.0]
        var got = _env_run[True](env, ctx, base, sc, W_SETTER)
        var plain_env = GEnv[False](ctx)
        var plain = _env_run[False](plain_env, ctx, base, none, W_SETTER)
        var tw0 = GEnv[False](ctx)
        _edit_model[False](tw0, ctx, 2.0, True)
        var twin0 = _env_run[False](tw0, ctx, base, none, W_SETTER)
        var tw1 = GEnv[False](ctx)
        _edit_model[False](tw1, ctx, 4.0, True)
        var twin1 = _env_run[False](tw1, ctx, base, none, W_SETTER)
        var d0 = _dist(got, twin0, 0, ROW)
        var d1 = _dist(got, twin1, ROW, 2 * ROW)
        var e0 = _dist(got, plain, 0, ROW)
        var e1 = _dist(got, plain, ROW, 2 * ROW)
        var nan = 0
        for i in range(len(got)):
            if got[i] != got[i]:
                nan += 1
        tl.check(nan == 0 and len(got) == NL * ROW, String(len(got)) + " values, " + String(nan) + " NaN")
        tl.check(d0[0] == 0, "lane 0 (crate x2) after " + String(ENV_STEPS) + " control steps: " + String(d0[0]) + " of " + String(ROW) + " values differ from its twin (max " + String(d0[1]) + ")")
        tl.check(d1[0] == 0, "lane 1 (crate x4): " + String(d1[0]) + " differ (max " + String(d1[1]) + ")")
        tl.check(e0[1] > 1e-3 and e1[1] > 1e-3, "against the unscaled model: max " + String(e0[1]) + " and " + String(e1[1]))

        print("[2] no scale is the plain env")
        var zero = _env_run[True](env, ctx, base, none, W_SETTER)
        var dz = _dist(zero, plain, 0, 2 * ROW)
        tl.check(dz[0] == 0, "LANE_MASS with nothing scaled: " + String(dz[0]) + " values differ from the plain env")

        print("[3] the solver's weights are live")
        var tw0n = GEnv[False](ctx)
        _edit_model[False](tw0n, ctx, 2.0, False)
        var twin0n = _env_run[False](tw0n, ctx, base, none, W_SETTER)
        var dn = _dist(got, twin0n, 0, ROW)
        tl.check(dn[0] > 0 and dn[1] > 1e-6, "lane 0 against a twin without the weights: " + String(dn[0]) + " differ, max " + String(dn[1]))

        print("[4] the weights from the device kernel")
        var by_kernel = _env_run[True](env, ctx, base, sc, W_KERNEL)
        var dk = _dist(by_kernel, got, 0, 2 * ROW)
        tl.check(dk[0] == 0, "scales written alone, then sync_lane_mass: " + String(dk[0]) + " values differ from the host setter's run")

        print("[5] the env's integrator: the device against the CPU")
        var s3: List[Float64] = [0.0, 3.0]
        var g3 = _raw_gpu(env, ctx, base, s3)
        var g0 = _raw_gpu(env, ctx, base, none)
        var c3 = _raw_cpu(env, base, s3)
        var c0 = _raw_cpu(env, base, none)
        for l in range(NL):
            var ee0 = _dist(g0, c0, l * ROW, (l + 1) * ROW)[1]
            var ee = _dist(g3, c3, l * ROW, (l + 1) * ROW)[1]
            var eff = _dist(c3, c0, l * ROW, (l + 1) * ROW)[1]
            var bar = max(10.0 * ee0, 1e-5)
            tl.check(ee <= bar, "lane " + String(l) + " after " + String(RAW_STEPS) + " steps: device vs CPU " + String(ee) + " (no scale " + String(ee0) + ", bar " + String(bar) + "), the scale's effect " + String(eff))

        print("\n===", tl.ok, "/", tl.total, "passed ===")
        if tl.ok != tl.total:
            raise Error("test_lane_mass_g1_gpu: " + String(tl.total - tl.ok) + " failed")
