"""The G1 pieces of FB-MEBE (`noeira/deep_agents/fb/g1_mebe.mojo`,
FB_MEBE_G1_PLAN.md §2 / §3).

  [1] B on ϑ_g: `G1BNetProj(s) == BFMBNet(s[ϑ])` bit for bit (CPU).
  [2] the suite: 22 tasks, 13 / 6 / 3 per family; every task's reward is
      1 at its own target state and drops when the state leaves it on an
      axis the task names, and does NOT drop on an axis it does not.
  [3] the fall kernel (device): a lane held down is reset exactly every
      `delay` steps, a tilted one too, an upright one never; the env's own
      time limit passes through on its step only; `_done` is written back.
  [4] (NVIDIA) on a 64-lane G1: the reset noise moves only the masked
      lanes' joints, within ±amp and inside the joint ranges; after random
      steps, `g1_reg_env_kernel` equals a host recomputation from the
      downloaded fields (feet orientation, limits and slide re-derived here;
      the contact scan is the env's `g1r_contacts`, shared — so this leg
      gates the kernel's plumbing and arithmetic, not the scan).

Run: pixi run -e nvidia mojo run -I . tests/deep_agents/test_fb_mebe_g1.mojo
"""

from std.math import sqrt, sin
from std.random import random_float64, seed
from std.sys import default_accelerator
from std.testing import assert_true
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.core.initializer import Xavier
from noeira.nn.core.param import ParamVisitor
from noeira.deep_agents.fb.bfm_towers import BFMBNet
from noeira.deep_agents.fb.kernels import _blocks
from noeira.deep_agents.fb.g1_mebe import (
    G1BNetProj,
    G1RegWeights,
    G1M_THETA, G1M_COL_VX, G1M_COL_VY, G1M_COL_WZ, G1M_COL_GX, G1M_COL_GY,
    G1M_COL_GZ, G1M_COL_H, G1M_SOFT_LIMIT,
    g1_cmd_suite,
    g1_cmd_reward_obs,
    g1_fall_kernel,
    g1_reset_noise_kernel,
    g1_reg_env_kernel,
)
from noeira.envs.robots import UnitreeG1Batched
from noeira.envs.robots.unitree_g1_xml import UNITREE_G1_OBS_DIM, ROOT_QPOS_SIZE
from noeira.envs.robots.unitree_g1_config import _projected_gravity
from noeira.envs.robots.unitree_g1_pd import (
    G1_N_DOF, g1_default_pos, g1_pos_lower, g1_pos_upper,
)
from noeira.envs.robots.unitree_g1_walk_rp_config import (
    g1r_contacts, G1R_L_ANKLE_ROLL, G1R_R_ANKLE_ROLL,
)
from noeira.physics3d.gpu.constants import CONTACT_SIZE, METADATA_SIZE

comptime OBS: Int = UNITREE_G1_OBS_DIM
comptime D: Int = 64
comptime HB: Int = 64
comptime N: Int = 8


struct _Grab(ParamVisitor):
    var vals: List[List[Scalar[DT]]]

    def __init__(out self):
        self.vals = List[List[Scalar[DT]]]()

    def visit[target: StaticString, NP: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var g = List[Scalar[DT]](capacity=NP)
        for i in range(NP):
            g.append(param.data[i])
        self.vals.append(g^)


struct _Put(ParamVisitor):
    var vals: List[List[Scalar[DT]]]
    var k: Int

    def __init__(out self, var vals: List[List[Scalar[DT]]]):
        self.vals = vals^
        self.k = 0

    def visit[target: StaticString, NP: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        assert_true(len(self.vals[self.k]) == NP, "param sizes differ")
        for i in range(NP):
            param.data[i] = self.vals[self.k][i]
        self.k += 1


def leg1() raises:
    print("[1] G1BNetProj == BFMBNet on the gathered ϑ ...")
    var COLS: List[Int] = [58, 59, 60, 64, 341, 342, 343, 434, 435, 436]
    var proj = G1BNetProj[OBS, D, HB].make["cpu", Xavier](None)
    var mlp = BFMBNet[G1M_THETA, D, HB].make["cpu", Xavier](None)
    var g0 = _Grab()
    proj.for_each_param["cpu"](g0, None)
    var put = _Put(g0.vals.copy())
    mlp.for_each_param["cpu"](put, None)
    assert_true(put.k == len(g0.vals), "param walks differ in length")
    var x = Tensor.alloc(N * OBS)
    for i in range(N * OBS):
        x.data[i] = Scalar[DT](random_float64() * 4.0 - 2.0)
    var xs = Tensor.alloc(N * G1M_THETA)
    for i in range(N):
        for k in range(G1M_THETA):
            xs.data[i * G1M_THETA + k] = x.data[i * OBS + COLS[k]]
    var y1 = Tensor()
    var y2 = Tensor()
    call_forward["cpu", N](proj, TensorRefs[1, MutAnyOrigin](x), y1, None)
    call_forward["cpu", N](mlp, TensorRefs[1, MutAnyOrigin](xs), y2, None)
    var diff = 0
    for i in range(N * D):
        if y1.data[i] != y2.data[i]:
            diff += 1
    print("      differing outputs:", diff, "/", N * D)
    assert_true(diff == 0, "projected B differs from the MLP on gathered columns")


def _row(
    vx: Float64, vy: Float64, wz: Float64, gx: Float64, gy: Float64, h: Float64
) -> List[Scalar[DT]]:
    var r = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    r[G1M_COL_VX] = Scalar[DT](vx)
    r[G1M_COL_VY] = Scalar[DT](vy)
    r[G1M_COL_WZ] = Scalar[DT](wz)
    r[G1M_COL_GX] = Scalar[DT](gx)
    r[G1M_COL_GY] = Scalar[DT](gy)
    r[G1M_COL_GZ] = Scalar[DT](-sqrt(max(0.0, 1.0 - gx * gx - gy * gy)))
    r[G1M_COL_H] = Scalar[DT](h)
    return r^


def leg2() raises:
    print("[2] the base-command suite ...")
    var s = g1_cmd_suite()
    var fam = List[Int](length=3, fill=0)
    for k in range(len(s)):
        fam[s[k].family] += 1
    print("      tasks", len(s), " per family", fam[0], fam[1], fam[2])
    assert_true(len(s) == 22, "suite size")
    assert_true(fam[0] == 13 and fam[1] == 6 and fam[2] == 3, "family sizes")
    var bad = 0
    for k in range(len(s)):
        ref c = s[k]
        var at = _row(c.vx, c.vy, c.wz, c.gx, c.gy, c.h)
        var r0 = g1_cmd_reward_obs(c, at, 0)
        if abs(r0 - 1.0) > 1e-6:
            print("      !!", c.name, "reward at target", r0)
            bad += 1
        # off on vx by 0.5: drops iff the task names velocity
        var off_v = _row(c.vx + 0.5, c.vy, c.wz, c.gx, c.gy, c.h)
        var rv = g1_cmd_reward_obs(c, off_v, 0)
        if c.use_vw != (rv < 0.5):
            print("      !!", c.name, "vx-off reward", rv, " use_vw", c.use_vw)
            bad += 1
        # off on h by 0.1: drops iff the task names height
        var off_h = _row(c.vx, c.vy, c.wz, c.gx, c.gy, c.h - 0.1)
        var rh = g1_cmd_reward_obs(c, off_h, 0)
        if c.use_h != (rh < 0.5):
            print("      !!", c.name, "h-off reward", rh, " use_h", c.use_h)
            bad += 1
        # off on tilt by 10°: every task names gravity
        var s10 = sin(10.0 * 3.141592653589793 / 180.0)
        var off_g = _row(c.vx, c.vy, c.wz, c.gx, c.gy + s10, c.h)
        var rg = g1_cmd_reward_obs(c, off_g, 0)
        if not (rg < 0.5):
            print("      !!", c.name, "tilt-off reward", rg)
            bad += 1
    assert_true(bad == 0, "suite rewards")


def leg3(ctx: DeviceContext) raises:
    print("[3] fall kernel: delay, tilt, time limit, write-back ...")
    comptime NL = 4
    comptime DELAY = 25
    var obs = ctx.enqueue_create_buffer[DT](NL * OBS)
    var h_obs = ctx.enqueue_create_host_buffer[DT](NL * OBS)
    for i in range(NL * OBS):
        h_obs[i] = Scalar[DT](0.0)
    for l in range(NL):
        h_obs[l * OBS + G1M_COL_H] = Scalar[DT](0.78)
        h_obs[l * OBS + G1M_COL_GZ] = Scalar[DT](-1.0)
    h_obs[1 * OBS + G1M_COL_H] = Scalar[DT](0.30)     # down
    h_obs[2 * OBS + G1M_COL_GZ] = Scalar[DT](-0.30)   # tilted past 60°
    ctx.enqueue_copy(obs, h_obs)
    var env_done = ctx.enqueue_create_buffer[DT](NL)
    var cnt = ctx.enqueue_create_buffer[DT](NL)
    var rec = ctx.enqueue_create_buffer[DT](NL)
    var nf = ctx.enqueue_create_buffer[DT](NL)
    cnt.enqueue_fill(Scalar[DT](0.0))
    nf.enqueue_fill(Scalar[DT](0.0))
    var h_ed = ctx.enqueue_create_host_buffer[DT](NL)
    var h_rec = ctx.enqueue_create_host_buffer[DT](NL)
    var h_back = ctx.enqueue_create_host_buffer[DT](NL)
    var h_nf = ctx.enqueue_create_host_buffer[DT](NL)
    var bad = 0
    for t in range(60):
        for l in range(NL):
            h_ed[l] = Scalar[DT](1.0) if (l == 3 and t == 3) else Scalar[DT](0.0)
        ctx.enqueue_copy(env_done, h_ed)
        ctx.enqueue_function[g1_fall_kernel[OBS, NL]](
            mptr(obs.unsafe_ptr()), mptr(env_done.unsafe_ptr()),
            mptr(cnt.unsafe_ptr()), mptr(rec.unsafe_ptr()), mptr(nf.unsafe_ptr()),
            Scalar[DT](0.45), Scalar[DT](-0.5), Int32(DELAY),
            grid_dim=1, block_dim=TPB,
        )
        ctx.enqueue_copy(h_rec, rec)
        ctx.enqueue_copy(h_back, env_done)
        ctx.synchronize()
        for l in range(NL):
            var want = False
            if l == 1 or l == 2:
                want = (t + 1) % DELAY == 0
            elif l == 3:
                want = t == 3
            var got = h_rec[l] > Scalar[DT](0.5)
            if got != want or h_back[l] != h_rec[l]:
                print("      !! lane", l, "step", t, "done", h_rec[l], "want", want, "back", h_back[l])
                bad += 1
    ctx.enqueue_copy(h_nf, nf)
    ctx.synchronize()
    print("      fall resets per lane:", h_nf[0], h_nf[1], h_nf[2], h_nf[3], "(want 0 2 2 0)")
    assert_true(bad == 0, "fall kernel schedule")
    assert_true(
        h_nf[0] == 0 and h_nf[1] == 2 and h_nf[2] == 2 and h_nf[3] == 0,
        "fall counts",
    )


def _rd(x: Scalar[DT]) -> Float64:
    return Float64(x)


def leg4(ctx: DeviceContext) raises:
    print("[4] G1 env: reset noise and the regularity terms ...")
    comptime NL = 64
    comptime EnvT = UnitreeG1Batched[NL]
    comptime NQ = EnvT.NQ
    comptime NB = EnvT.NBODY
    comptime MC = EnvT.MC
    var env = EnvT(ctx)
    env.reset_batch[NL](Optional(ctx), UInt64(3))
    # reset the EVEN lanes only, with noise
    var h_done = ctx.enqueue_create_host_buffer[DT](NL)
    for l in range(NL):
        h_done[l] = Scalar[DT](1.0) if l % 2 == 0 else Scalar[DT](0.0)
    ctx.enqueue_copy(env._done, h_done)
    env.selective_reset_batch[NL](Optional(ctx), UInt64(4))
    comptime AMP = 0.1
    ctx.enqueue_function[g1_reset_noise_kernel[NQ, NL]](
        mptr(env.d.qpos.dev.value().unsafe_ptr()), mptr(env._reset_mask.unsafe_ptr()),
        Scalar[DT](AMP), UInt64(99),
        grid_dim=_blocks(NL), block_dim=TPB,
    )
    env._run_fields_fk(ctx)
    env._run_fields_vel(ctx)
    env._extract_obs_only(ctx)
    var h_q = ctx.enqueue_create_host_buffer[DT](NL * NQ)
    ctx.enqueue_copy(h_q, env.d.qpos.dev.value())
    ctx.synchronize()
    var bad = 0
    var moved = 0
    for l in range(NL):
        for i in range(G1_N_DOF):
            var q = Float64(h_q[l * NQ + ROOT_QPOS_SIZE + i])
            var e = q - g1_default_pos(i)
            if l % 2 == 0:
                if abs(e) > AMP + 1e-6 or q < g1_pos_lower(i) - 1e-6 or q > g1_pos_upper(i) + 1e-6:
                    bad += 1
                if abs(e) > 1e-4:
                    moved += 1
            elif abs(e) > 1e-6:
                bad += 1   # an unmasked lane must sit at the stand pose
    print("      masked joints moved", moved, "/", (NL // 2) * G1_N_DOF, "  violations", bad)
    assert_true(bad == 0, "reset noise outside its lanes / band / range")
    assert_true(moved > (NL // 2) * G1_N_DOF * 9 // 10, "reset noise did not move the joints")

    # random steps, then the kernel against the host
    var h_act = ctx.enqueue_create_host_buffer[DT](NL * 29)
    for _ in range(30):
        for i in range(NL * 29):
            h_act[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        ctx.enqueue_copy(env._action, h_act)
        env.step_batch[NL](Optional(ctx), UInt64(5))
    var r_env = ctx.enqueue_create_buffer[DT](NL)
    var w = G1RegWeights.default()
    ctx.enqueue_function[g1_reg_env_kernel[NL, NQ, NB, MC]](
        env.d.qpos.lt["gpu", type_of(env.d).L_QPOS](),
        env.d.xquat.lt["gpu", type_of(env.d).L_B4](),
        env.d.xvel.lt["gpu", type_of(env.d).L_B3](),
        env.d.contacts.lt["gpu", type_of(env.d).L_CONTACTS](),
        env.d.meta.lt["gpu", type_of(env.d).L_META](),
        mptr(r_env.unsafe_ptr()),
        Scalar[DT](w.undesired), Scalar[DT](w.feet_orient),
        Scalar[DT](w.limits), Scalar[DT](w.slide),
        grid_dim=_blocks(NL), block_dim=TPB,
    )
    comptime NCON = MC * CONTACT_SIZE
    var hq = ctx.enqueue_create_host_buffer[DT](NL * NQ)
    var hxq = ctx.enqueue_create_host_buffer[DT](NL * NB * 4)
    var hxv = ctx.enqueue_create_host_buffer[DT](NL * NB * 3)
    var hc = ctx.enqueue_create_host_buffer[DT](NL * NCON)
    var hm = ctx.enqueue_create_host_buffer[DT](NL * METADATA_SIZE)
    var hr = ctx.enqueue_create_host_buffer[DT](NL)
    ctx.enqueue_copy(hq, env.d.qpos.dev.value())
    ctx.enqueue_copy(hxq, env.d.xquat.dev.value())
    ctx.enqueue_copy(hxv, env.d.xvel.dev.value())
    ctx.enqueue_copy(hc, env.d.contacts.dev.value())
    ctx.enqueue_copy(hm, env.d.meta.dev.value())
    ctx.enqueue_copy(hr, r_env)
    ctx.synchronize()
    var ct = LayoutTensor[DT, Layout.row_major(NL, NCON), MutAnyOrigin](mptr(hc.unsafe_ptr()))
    var mt = LayoutTensor[DT, Layout.row_major(NL, METADATA_SIZE), MutAnyOrigin](mptr(hm.unsafe_ptr()))
    var worst = 0.0
    var nonzero = 0
    var n_und = 0
    var n_slide = 0
    for l in range(NL):
        var fc = g1r_contacts[DT, NL, MC](ct, mt, l)
        var fo = 0.0
        for f in range(2):
            var b = G1R_L_ANKLE_ROLL if f == 0 else G1R_R_ANKLE_ROLL
            var o = (l * NB + b) * 4
            var g = _projected_gravity(
                _rd(hxq[o + 3]), _rd(hxq[o]), _rd(hxq[o + 1]), _rd(hxq[o + 2])
            )
            fo += g[0] * g[0] + g[1] * g[1]
        var lim = 0.0
        for i in range(G1_N_DOF):
            var q = _rd(hq[l * NQ + ROOT_QPOS_SIZE + i])
            var c = 0.5 * (g1_pos_lower(i) + g1_pos_upper(i))
            var r = 0.5 * (g1_pos_upper(i) - g1_pos_lower(i)) * G1M_SOFT_LIMIT
            if q < c - r:
                lim += (c - r) - q
            elif q > c + r:
                lim += q - (c + r)
        var sl = 0.0
        for f in range(2):
            var on = fc.left if f == 0 else fc.right
            if on:
                var b = G1R_L_ANKLE_ROLL if f == 0 else G1R_R_ANKLE_ROLL
                var o = (l * NB + b) * 3
                sl += sqrt(_rd(hxv[o]) ** 2 + _rd(hxv[o + 1]) ** 2)
                n_slide += 1
        n_und += fc.undesired
        var want = -(
            w.undesired * Float64(fc.undesired) + w.feet_orient * fo
            + w.limits * lim + w.slide * sl
        )
        var got = Float64(hr[l])
        var e = abs(got - want) / max(1.0, abs(want))
        if e > worst:
            worst = e
        if abs(got) > 1e-9:
            nonzero += 1
    print(
        "      worst rel err", worst, "  nonzero lanes", nonzero, "/", NL,
        "  undesired contacts", n_und, "  feet in contact", n_slide,
    )
    assert_true(worst < 1e-4, "g1_reg_env_kernel differs from the host recomputation")
    assert_true(nonzero > NL // 2, "the regularity terms are all zero — vacuous leg")


def main() raises:
    seed(20261013)
    leg1()
    leg2()
    with DeviceContext() as ctx:
        leg3(ctx)
        comptime if default_accelerator().is_nvidia_gpu():
            leg4(ctx)
        else:
            print("[4] skipped (the G1 batched env is NVIDIA-only)")
    print("ALL PASS — FB-MEBE G1 pieces")
